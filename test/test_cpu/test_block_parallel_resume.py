# Copyright (c) 2025 Intel Corporation
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

"""Tests for block-parallel per-block resume (results + chain checkpoints)."""

import os
import tempfile
import unittest

import torch

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402


class TestBlockResults(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            layers = {"model.layers.3.mlp.up_proj": {"scale": torch.tensor([1.5]), "zp": torch.tensor([2])}}
            block_parallel.save_block_results(d, "model.layers.3", layers)
            self.assertTrue(block_parallel.has_block_results(d, "model.layers.3"))
            self.assertFalse(block_parallel.has_block_results(d, "model.layers.4"))
            merged = block_parallel.load_all_block_results(d)
            self.assertIn("model.layers.3.mlp.up_proj", merged)

    def test_two_blocks_merge(self):
        with tempfile.TemporaryDirectory() as d:
            for b, v in [("model.layers.0", 0.5), ("model.layers.1", 1.5)]:
                block_parallel.save_block_results(d, b, {f"{b}.q": {"scale": torch.tensor([v]), "zp": None}})
            merged = block_parallel.load_all_block_results(d)
            self.assertEqual(len(merged), 2)

    def test_coverage_check(self):
        with tempfile.TemporaryDirectory() as d:
            all_blocks = [["model.layers.0", "model.layers.1"], ["model.layers.2"]]
            missing = block_parallel.missing_result_blocks(d, all_blocks)
            self.assertEqual(missing, ["model.layers.0", "model.layers.1", "model.layers.2"])
            block_parallel.save_block_results(
                d, "model.layers.1", {"x": {"scale": torch.tensor([1.0]), "zp": None}}
            )
            missing = block_parallel.missing_result_blocks(d, all_blocks)
            self.assertEqual(missing, ["model.layers.0", "model.layers.2"])


class TestChainCheckpoints(unittest.TestCase):
    def test_roundtrip_dtype_restored(self):
        with tempfile.TemporaryDirectory() as d:
            hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
            block_parallel.save_chain_state(d, group=0, block_idx=5, hidden=hidden)
            restored = block_parallel.load_chain_state(d, group=0, block_idx=5, device="cpu")
            self.assertEqual(restored.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(restored, hidden))
            self.assertIsNone(block_parallel.load_chain_state(d, group=0, block_idx=6, device="cpu"))

    def test_keys_are_group_scoped(self):
        with tempfile.TemporaryDirectory() as d:
            block_parallel.save_chain_state(d, group=2, block_idx=1, hidden=torch.zeros(1))
            self.assertIsNone(block_parallel.load_chain_state(d, group=1, block_idx=1, device="cpu"))


class TestResumeDirReuse(unittest.TestCase):
    """AR_RESUME_DIR (the serial flag) selects parallel resume storage."""

    def test_none_without_resume_dir(self):
        os.environ.pop("AR_RESUME_DIR", None)
        import importlib

        from auto_round import envs

        old = envs.AR_RESUME_DIR
        os.environ["AR_RESUME_DIR"] = ""
        try:
            importlib.reload(envs)
            self.assertIsNone(block_parallel.parallel_results_dir())
        finally:
            if old:
                os.environ["AR_RESUME_DIR"] = old
            else:
                os.environ.pop("AR_RESUME_DIR", None)
            importlib.reload(envs)

    def test_signature_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(block_parallel.signature_matches(d, "sig-A"))  # absent = first run
            block_parallel.save_signature(d, "sig-A")
            self.assertTrue(block_parallel.signature_matches(d, "sig-A"))
            self.assertFalse(block_parallel.signature_matches(d, "sig-B"))


class TestSharedNonblocks(unittest.TestCase):
    def test_save_apply_roundtrip_excludes_blocks(self):
        import torch
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
                self.head = nn.Linear(4, 8)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nonblocks.pt")
            model = Tiny()
            n = block_parallel.save_shared_nonblocks(model, ["blocks"], path)
            # embed.weight + head.weight + head.bias saved; blocks excluded
            self.assertEqual(n, 3)
            # fresh model on meta: apply restores the tensors
            fresh = Tiny()
            for param in fresh.parameters():
                param.data = torch.empty(0, device="meta")
            applied = block_parallel.apply_shared_nonblocks(fresh, ["blocks"], path)
            self.assertEqual(applied, 3)
            self.assertTrue(torch.equal(fresh.embed.weight, model.embed.weight))
            self.assertTrue(fresh.blocks[0].weight.device.type == "meta")


class TestSharedInputs(unittest.TestCase):
    def test_payload_roundtrip(self):
        import torch

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calib_inputs.pt")
            payload = {
                "inputs": {"model.layers.0": {"input_ids": torch.randn(2, 3, 4)}},
                "input_ids": [torch.ones(1, 5, dtype=torch.long)],
                "batch_size": 2,
                "seqlen": 256,
                "batch_dim": 0,
                "is_only_supported_bs1": False,
                "orig_batch_size": 4,
                "block_forward_batch_size": 2,
                "ga_steps": [8, None],
            }
            block_parallel.save_shared_inputs(payload, path)
            back = block_parallel.load_shared_inputs(path)
            self.assertEqual(back["batch_size"], 2)
            self.assertEqual(back["ga_steps"], [8, None])
            self.assertTrue(torch.equal(back["inputs"]["model.layers.0"]["input_ids"], payload["inputs"]["model.layers.0"]["input_ids"]))


if __name__ == "__main__":
    unittest.main()
