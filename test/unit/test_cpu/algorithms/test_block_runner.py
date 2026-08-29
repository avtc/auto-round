# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from auto_round.algorithms.block_runner import BlockForwardRunner


class GlmMoeDsaDecoderLayer(torch.nn.Module):
    def __init__(self, shared=False):
        super().__init__()
        self.shared = shared

    def forward(self, hidden_states, prev_topk_indices=None):
        if self.shared:
            assert prev_topk_indices is not None
            expected_indices = (hidden_states[..., :1] - 1).to(torch.long)
            torch.testing.assert_close(prev_topk_indices, expected_indices)
            topk_indices = prev_topk_indices
        else:
            topk_indices = hidden_states[..., :1].to(torch.long)
        return hidden_states + 1, topk_indices


class TupleOutputBlock(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1, hidden_states + 2


def test_glm_dsa_topk_indices_propagate_between_blocks():
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    inputs = [torch.full((1, 3, 2), value, dtype=torch.float32) for value in range(3)]

    reference_outputs = runner(GlmMoeDsaDecoderLayer(), inputs, {})
    next_inputs = runner.last_output_dict

    assert next_inputs is not None
    assert set(next_inputs) == {"hidden_states", "prev_topk_indices"}
    assert next_inputs["hidden_states"] is reference_outputs

    shared_outputs = runner(GlmMoeDsaDecoderLayer(shared=True), next_inputs, {})

    assert len(shared_outputs) == len(inputs)
    assert runner.last_output_dict is not None


def test_unregistered_tuple_output_keeps_first_tensor_behavior():
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    inputs = [torch.zeros((1, 2, 2)), torch.ones((1, 2, 2))]

    outputs = runner(TupleOutputBlock(), inputs, {})

    assert runner.last_output_dict is None
    for output, input_tensor in zip(outputs, inputs):
        torch.testing.assert_close(output, input_tensor + 1)


def test_indexed_single_sample_forward_preserves_one_batch_dimension():
    runner = BlockForwardRunner(
        batch_dim=0,
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=True,
        amp_dtype=torch.bfloat16,
    )
    sample = torch.randn(1, 3, 4)

    output = runner(torch.nn.Identity(), [sample], {}, indices=torch.tensor([0]))

    assert output.shape == sample.shape
    torch.testing.assert_close(output, sample)


def test_indexed_diffusion_outputs_preserve_batch_dimension():
    class FluxTransformerBlock(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states + 1, hidden_states + 2

    runner = BlockForwardRunner(
        batch_dim=0,
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=True,
        amp_dtype=torch.bfloat16,
        is_diffusion=True,
    )
    sample = torch.randn(1, 3, 4)

    output = runner(
        FluxTransformerBlock(),
        {"hidden_states": [sample]},
        {},
        indices=torch.tensor([0]),
    )

    assert output.shape == sample.shape
    assert runner.last_output_dict["encoder_hidden_states"].shape == sample.shape
    assert runner.last_output_dict["hidden_states"].shape == sample.shape


def test_outputs_are_moved_to_cache_device_before_next_batch(monkeypatch):
    events = []

    class TrackingTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value, batch_index):
            result = torch.Tensor._make_subclass(cls, value, require_grad=False)
            result.batch_index = batch_index
            return result

        def to(self, *args, **kwargs):
            events.append(("move", self.batch_index))
            return super().to(*args, **kwargs)

    runner = BlockForwardRunner(batch_size=1, device="cpu", cache_device="cpu", amp=False)
    call_count = 0

    def fake_forward_one_batch(_block, batch_inputs, _batch_others):
        nonlocal call_count
        if call_count:
            assert ("move", call_count - 1) in events
        output = TrackingTensor(batch_inputs, call_count)
        call_count += 1
        return output

    monkeypatch.setattr(runner, "_forward_one_batch", fake_forward_one_batch)
    inputs = [torch.full((1, 2), value, dtype=torch.float32) for value in range(3)]

    outputs = runner(torch.nn.Identity(), inputs, {})

    assert len(outputs) == len(inputs)
    assert call_count == len(inputs)


def test_select_batch_accepts_host_int_indices():
    """Host-int indices select identically to tensor indices (no per-element fences)."""
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    inputs = [torch.full((1, 2, 2), v, dtype=torch.float32) for v in range(4)]

    tensor_sel = runner.select_batch(inputs, {}, torch.tensor([3, 1]))
    host_sel = runner.select_batch(inputs, {}, [3, 1])
    torch.testing.assert_close(tensor_sel[0], host_sel[0])

    stacked = torch.stack(inputs).squeeze(1)  # tensor input -> index_select path
    torch.testing.assert_close(
        runner.select_batch(stacked, {}, torch.tensor([3, 1]))[0],
        runner.select_batch(stacked, {}, [3, 1])[0],
    )

    runner.shared_cache_keys = ("shared",)
    sel = runner.select_batch({"shared": ["x", "y", "z", "w"]}, {}, [3])
    assert sel[0]["shared"] == "w"  # single-index selection picks val[3]


def test_select_batch_host_indices_cover_tensor_others_branch():
    """Bare-tensor others values (attention-mask style) take index_select."""
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    others = {"attention_mask": torch.arange(8, dtype=torch.float32).reshape(4, 2)}
    sel = runner.select_batch([torch.zeros(1, 2, 2)] * 4, others, [3, 1])
    torch.testing.assert_close(sel[1]["attention_mask"], others["attention_mask"][[3, 1]])


class TestSerialDeviceSchedule:
    """T8a: per-block schedule precompute + pinned device-index staging."""

    def test_schedule_matches_lazy_sampler_across_reshuffle(self):
        import random

        from auto_round.compressors.utils import IndexSampler

        random.seed(1234)
        lazy = IndexSampler(nsamples=20, batch_size=6)  # forces reshuffles
        want = [list(lazy.next_batch()) for _ in range(9)]
        random.seed(1234)
        eager = IndexSampler(nsamples=20, batch_size=6)
        pre = [list(eager.next_batch()) for _ in range(9)]
        assert pre == want  # identical RNG stream incl. reshuffle boundaries

    def test_env_gate_default_on_opt_out(self, monkeypatch):
        from auto_round import envs

        monkeypatch.delenv("AR_TUNE_SERIAL_DEVICE_SCHEDULE", raising=False)
        assert envs.AR_TUNE_SERIAL_DEVICE_SCHEDULE is True
        monkeypatch.setenv("AR_TUNE_SERIAL_DEVICE_SCHEDULE", "0")
        assert envs.AR_TUNE_SERIAL_DEVICE_SCHEDULE is False

    def test_staged_index_tensor_round_trip(self):
        import torch

        rows = [[3, 1, 4], [5, 9, 2], [6, 5, 3]]
        host = torch.tensor(rows, dtype=torch.long)  # pin_memory only under CUDA
        dev_buf = torch.zeros(3, dtype=torch.long)
        for i in range(3):
            dev_buf.copy_(host[i], non_blocking=True)
            assert dev_buf.tolist() == rows[i]


class TestSelectBatchHostIndices:
    """T8a: tensor indices + host row must equal legacy host-int selection."""

    def _runner(self):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        return BlockForwardRunner.__new__(BlockForwardRunner)

    def test_mixed_dict_tensor_plus_host_equals_legacy(self):
        import torch

        r = self._runner()
        r.batch_dim = 0
        r.shared_cache_keys = {"past"}
        inputs = {
            "hidden": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "past": [torch.full((2, 2), float(i)) for i in range(6)],
        }
        others = {"positional_inputs": None, "mask": [torch.ones(2, 4) * i for i in range(6)]}
        idx = torch.tensor([4, 1, 3])
        legacy_inputs, legacy_others = r._select_batch(
            {"hidden": inputs["hidden"].clone(), "past": list(inputs["past"])},
            {"positional_inputs": None, "mask": list(others["mask"])},
            [4, 1, 3],
        )
        new_inputs, new_others = r._select_batch(inputs, others, idx, host_indices=[4, 1, 3])
        assert torch.equal(new_inputs["hidden"], legacy_inputs["hidden"])
        assert torch.equal(new_inputs["past"], legacy_inputs["past"])
        assert torch.equal(new_others["mask"], legacy_others["mask"])
        # tensor path without host row must still work for pure-tensor inputs
        t_in = {"hidden": inputs["hidden"]}
        t_out, _ = r._select_batch(t_in, {"positional_inputs": None}, idx)
        assert torch.equal(t_out["hidden"], legacy_inputs["hidden"])
