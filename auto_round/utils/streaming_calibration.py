# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Streaming calibration cache: per-block FP inputs without a full model load.

Runs before the zero-shot quantization loop when the activation chain engages:
calibration rows are pushed through the model one block at a time (blocks
streamed onto the compute device, hidden states chained block-to-block), and
each block's FP input tensors plus the forward kwargs are cached on the host -
the same structure the data-driven calibrator produces. The quantization loop
then replays the cached inputs through each block via the standard
``compress_block`` path: transforms (rotation, smoothing) run first and the
imatrix hooks fire on the replayed inputs, exactly matching the data-driven
semantics for identical rows.

Scope: block-local modules only. Modules outside the streamed blocks (e.g.
lm_head) receive no cached inputs; the quantizer falls back to unweighted
search there.

Memory: the cache holds one tensor per row per block on the host
(rows x blocks x seqlen x hidden). ``nsamples`` bounds it -
the imatrix is a column statistic, so a modest row count is statistically
sufficient.
"""

import inspect

import torch

from auto_round.logger import logger


def build_causal_attention_mask(input_ids):
    """4D boolean attention mask (True = allowed) combining the causal
    structure with the data-driven calibration key mask (all ones, trailing
    repeated tokens masked, last position always masked - mirrors
    calibration/llm.py and the model-level causal-mask preparation)."""
    seq_len = input_ids.shape[-1]
    mask_2d = torch.ones_like(input_ids, dtype=torch.long)
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        last_token = input_ids[i, -1]
        j = seq_len - 2
        repeated = False
        while j >= 0 and input_ids[i, j] == last_token:
            repeated = True
            mask_2d[i, j] = 0
            j -= 1
        if repeated:
            mask_2d[i, -1] = 0
    mask_2d[:, -1] = 0
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return causal[None, None] & (mask_2d != 0)[:, None, None, :]


def mask_form_candidates(key_mask_2d):
    """The three mask forms decoder layers accept, most-native first.

    1. 4D bool (True = allowed), causal + key mask: what llama-style models
       pass into their decoder layers after model-level preparation.
    2. 2D float padding mask: what GDN/hybrid models pass down (their layers
       build causal structure internally; the linear-attention path
       multiplies the 2D mask against the hidden states directly).
    3. 4D float additive (allowed = 0.0, masked = -inf): the sdpa convention,
       accepted by layers that feed the mask straight into scaled_dot_product_attention.
    """
    seq_len = key_mask_2d.shape[-1]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    allowed = causal[None, None] & (key_mask_2d != 0.0)[:, None, None, :]
    additive = torch.zeros(allowed.shape, dtype=torch.float32)
    additive.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    return [("4d_bool", allowed), ("2d_float", key_mask_2d), ("4d_additive", additive)]


def resolve_chain_mask_form(block, fp_row, key_mask_2d, input_others, preferred=None):
    """Probe which attention-mask form this model's decoder layers accept.

    Runs one tiny no-grad forward of the block per candidate form and
    returns the winning form name. Models can MIX conventions per block
    (e.g. GDN linear-attention blocks take the 2D padding mask while
    full-attention blocks of the same model take the 4D form), so each block
    is probed; ``preferred`` tries the previous block's form first (block
    types come in runs, so most probes hit immediately).
    """

    def _to_dev(v, dev):
        if isinstance(v, torch.Tensor):
            return v.to(dev)
        if isinstance(v, (list, tuple)):
            packed = type(v)
            return packed(_to_dev(x, dev) for x in v)
        return v

    row_others = {k: (v[0] if isinstance(v, list) else v) for k, v in input_others.items()}
    row_others["use_cache"] = False
    row_others["past_key_values"] = None
    dev = next((p.device for p in block.parameters() if p.device.type != "meta"), torch.device("cpu"))
    row_others = {k: _to_dev(v, dev) for k, v in row_others.items()}
    fp_row = fp_row[:1].to(dev)
    candidates = mask_form_candidates(key_mask_2d)
    if preferred is not None:
        order = [c for c in candidates if c[0] == preferred] + [c for c in candidates if c[0] != preferred]
    else:
        order = candidates
    last_err = None
    with torch.no_grad():
        for name, mask in order:
            probe_others = dict(row_others)
            probe_others["attention_mask"] = mask.to(dev)
            try:
                block(fp_row, **probe_others)
                return name
            except Exception as e:  # noqa: BLE001  shape/dtype/type errors are the signal
                last_err = e
                continue
    raise ValueError(
        "streaming calibration could not determine this model's attention-mask "
        "convention: the first decoder block rejected every candidate form "
        f"(last error: {last_err})."
    )


def materialize_mask_form(key_mask_2d, form):
    """The per-row mask tensor for the resolved form."""
    for name, mask in mask_form_candidates(key_mask_2d):
        if name == form:
            return mask
    raise ValueError(f"unknown mask form {form!r}")


def _find_embedding(model, streamer):
    """The token-embedding module: checkpoint-backed, preferring the config's
    vocab size, else the largest candidate.

    VL-capable checkpoints (e.g. qwen3_5) carry small positional Embeddings in
    the vision tower that iterate FIRST; taking the first match compared
    248k-token calibration ids against a 2304-row table. The vocab guard then
    (correctly) failed the run instead of letting an OOB gather poison CUDA.
    """
    cfg = getattr(model, "config", None)
    inner = getattr(cfg, "text_config", None)
    if inner is not None and getattr(inner, "vocab_size", None):
        cfg = inner  # VL wrapper: the token vocab lives in text_config
    want = getattr(cfg, "vocab_size", None)

    best_name, best_mod, best_n = None, None, -1
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and f"{name}.weight" in streamer.weight_map:
            n = module.num_embeddings if hasattr(module, "num_embeddings") else module.weight.shape[0]
            if want is not None and n == want:
                return name, module
            if n > best_n:
                best_name, best_mod, best_n = name, module, n
    return best_name, best_mod


def _ensure_real_rotary(rotary, cfg, device):
    """Return a rotary module with real tensors on *device*.

    A meta-built model leaves computed buffers such as ``inv_freq`` on meta;
    meta ops are lazy, so a forward through such a module silently produces
    meta cos/sin that detonate at the first real consumer. Rebuild the module
    on the device when any of its tensors are meta; move it when real.
    """
    tensors = list(rotary.parameters()) + list(rotary.buffers())
    if not any(t.device.type == "meta" for t in tensors):
        return rotary.to(device)

    text_cfg = getattr(cfg, "text_config", None) or cfg
    cls = type(rotary)
    candidates = []
    if hasattr(text_cfg, "rope_theta"):
        candidates.append(lambda: cls(config=text_cfg))
    inv_freq = getattr(rotary, "inv_freq", None)
    if inv_freq is not None and inv_freq.device.type == "meta" and inv_freq.ndim == 1:
        d = int(inv_freq.shape[0]) * 2
        base = getattr(rotary, "base", None) or getattr(text_cfg, "rope_theta", 10000.0)
        max_pos = getattr(text_cfg, "max_position_embeddings", 262144)

        def _manual():
            import torch.nn as nn

            mod = cls.__new__(cls)  # bypass __init__ device/context quirks
            nn.Module.__init__(mod)
            freqs = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
            mod.register_buffer("inv_freq", freqs, persistent=False)
            mod.base = base
            mod.max_seq_len_cached = max_pos
            return mod

        if getattr(text_cfg, "rope_scaling", None) or getattr(text_cfg, "rope_parameters", None):
            candidates.clear()  # manual formula ignores rope scaling: unsafe
            candidates.append(lambda: cls(config=text_cfg))
        else:
            candidates.append(_manual)
    for make in candidates:
        try:
            fresh = make().to(device)
            fresh_tensors = list(fresh.parameters()) + list(fresh.buffers())
            if not any(t.device.type == "meta" for t in fresh_tensors):
                return fresh
        except Exception:  # try the next construction pattern
            continue
    raise RuntimeError(
        "rotary_emb has meta tensors and could not be rebuilt on the device; "
        "construct it manually for this architecture"
    )


def _find_model_rotary(model, cfg, device):
    """Locate the TEXT model's rotary module (outside the blocks) and ensure it
    holds real tensors; returns the module or None.

    VL-capable checkpoints carry a vision rotary whose ``forward(x, dim)``
    signature differs from the text rotary's ``forward(x, position_ids)``; it
    iterates first in ``named_modules`` and must not be picked (same trap as
    ``_find_embedding`` and the vision position table)."""
    base = getattr(model, "model", model)
    # VL composites keep the text backbone under .language_model (sometimes
    # nested one .model deeper); resolve it before touching any fallback scan
    for attr in ("language_model",):
        nxt = getattr(base, attr, None)
        if nxt is not None:
            base = nxt
    rotary = getattr(base, "rotary_emb", None) or getattr(getattr(base, "model", None), "rotary_emb", None)
    if rotary is None:
        best, best_score = None, -1
        for name, mod in model.named_modules():
            leaf = not list(mod.children())
            if not leaf or "rotary" not in type(mod).__name__.lower():
                continue
            score = 0
            lname = name.lower()
            if "visio" in lname or "visual" in lname:
                score -= 4
            try:
                params = list(inspect.signature(mod.forward).parameters)[1:]
            except (TypeError, ValueError):
                params = []
            if "position_ids" in params:
                score += 2  # text rotary convention: forward(x, position_ids)
            if score > best_score:
                best, best_score = mod, score
        rotary = best
    if rotary is None:
        return None
    return _ensure_real_rotary(rotary, cfg, device)


def _normalize_rows(dataset, tokenizer, seqlen, seed=42, nsamples=128, bs=1):
    """Flatten the calibration dataset into a list of input_ids batches
    (tensors), applying the data-driven skip rule (shorter than seqlen) and an
    optional row cap.

    A string dataset follows the data-driven protocol (pile-10k, shuffle
    seed 42) tokenized with the MODEL's tokenizer, so ids are in-vocab by
    construction."""
    rows = []
    if isinstance(dataset, str):
        from auto_round.calib_dataset import get_dataloader

        dataset = get_dataloader(tokenizer, seqlen, dataset.replace(" ", ""), seed=seed, bs=bs, nsamples=nsamples)
    for data in dataset:
        if data.__class__.__name__ == "BatchEncoding":
            data = data.data
        if isinstance(data, torch.Tensor):
            input_ids = data
        elif isinstance(data, (tuple, list)):
            input_ids = data[0]
        else:
            if "input_ids" not in data:
                continue
            input_ids = data["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"] if isinstance(input_ids, dict) else input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[-1] < seqlen:
            continue
        rows.append(input_ids)
    # nsamples caps list datasets too (the data-driven calibrator stops at
    # nsamples for every dataset type; the chain matches that semantics).
    if nsamples and len(rows) > nsamples:
        rows = rows[:nsamples]
        logger.info("[stream_calibration] capping calibration rows to %d (nsamples)", nsamples)
    return rows


def _check_ids_in_vocab(rows, vocab):
    """Fail with an actionable message before a CUDA gather assert poisons the
    process (rows tokenized with a different model's tokenizer)."""
    top = max(int(max(int(r.max()), 0)) for r in rows) if rows else 0
    if top >= vocab:
        raise ValueError(
            f"stream_calibration: calibration token id {top} exceeds the embedding vocab ({vocab}) - "
            "the rows were tokenized with a different model's tokenizer; rebuild them "
            "with this model's tokenizer"
        )


def prepare_streaming_calibration(
    model, streamer, dataset, device, seqlen, tokenizer=None, first_block=None, nsamples=128
):
    """Initialize the streaming calibration chain.

    Embeds the calibration rows once and builds the shared per-row forward
    kwargs (4D boolean causal attention masks with the data-driven key-mask
    rule, position ids, rotary position embeddings) in the exact format the
    data-driven calibrator captures. The quantization loop then chains blocks:
    each block's ``compress_block`` reference output becomes the next block's
    FP inputs, matching the data-driven semantics (transforms applied before
    the replay that collects activation statistics).

    Returns ``(fp_inputs, input_others, summary)``: the initial per-row hidden
    states, the shared kwargs cache, and a small summary dict.
    """
    rows = _normalize_rows(dataset, tokenizer, seqlen, nsamples=nsamples)
    if not rows:
        raise ValueError("stream_calibration: no usable calibration rows (all shorter than seqlen?)")
    logger.info("[stream_calibration] chaining %d calibration rows (nsamples=%d)", len(rows), nsamples)

    embed_name, embed_mod = _find_embedding(model, streamer)
    if embed_mod is None:
        raise RuntimeError("stream_calibration: no checkpoint-backed embedding module found")
    streamer.load_module_(embed_mod, embed_name, device=device)
    vocab = embed_mod.num_embeddings if hasattr(embed_mod, "num_embeddings") else embed_mod.weight.shape[0]
    _check_ids_in_vocab(rows, vocab)
    with torch.no_grad():
        fp_inputs = [embed_mod(ids.to(device)).cpu() for ids in rows]
    embed_mod.to("meta")

    cfg = getattr(model, "config", None)
    rotary = _find_model_rotary(model, cfg, device)

    # the data-driven pipeline replays a float32 0/1 mask (bool captured, cast
    # during preprocessing); sdpa treats it additively. Match the dtype for
    # statistics parity with the data-driven path.
    # Canonical per-row 2D key mask (1.0 = keep, 0.0 = masked), the same form
    # the ordinary calibration path feeds model.forward: all ones, trailing
    # repeated tokens masked, last position always masked (see
    # calibration/llm.py). The form each decoder layer actually consumes is
    # model-dependent (llama-style layers receive a prepared 4D bool mask;
    # GDN linear-attention layers consume the 2D padding mask directly), so
    # the loop resolves it once by probing the first block (see
    # resolve_chain_mask_form) and materializes the winning form.
    masks = []
    for ids in rows:
        m = torch.ones(ids.shape, dtype=torch.float32, device=ids.device)
        last_token = ids[:, -1]
        j = ids.shape[-1] - 2
        for b in range(ids.shape[0]):
            repeated = False
            while j >= 0 and ids[b, j] == last_token[b]:
                repeated = True
                m[b, j] = 0.0
                j -= 1
            if repeated:
                m[b, -1] = 0.0
        m[:, -1] = 0.0
        masks.append(m.cpu())
    input_others = {
        "attention_mask": masks,
        "use_cache": False,
        "past_key_values": None,
    }

    # introspect the first streamed block's signature to know which positional
    # kwargs to supply; every block of a model shares the same layout here
    params = inspect.signature(first_block.forward).parameters if first_block is not None else {}
    if "position_ids" in params:
        input_others["position_ids"] = [torch.arange(ids.shape[-1]).unsqueeze(0) for ids in rows]
    if "position_embeddings" in params and rotary is not None:
        pe_list = []
        for ids in rows:
            pos = torch.arange(ids.shape[-1], device=device).unsqueeze(0)
            cos, sin = rotary(torch.zeros(1, ids.shape[-1], device=device, dtype=torch.float32), pos)
            pe_list.append((cos.cpu(), sin.cpu()))
        input_others["position_embeddings"] = pe_list

    # keep the raw token ids: SignRound's loss mask consumes them per block
    summary = {"rows": len(rows), "token_ids": rows, "keymask_2d": masks}
    return fp_inputs, input_others, summary
