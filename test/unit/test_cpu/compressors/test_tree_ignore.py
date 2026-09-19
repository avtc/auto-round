# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The predictor-tree stage applies the SAME export ignore list as the
completion pass, so run-quantized and completer-quantized trees produce
identical artifact shapes."""

import inspect

from auto_round.utils.missing_tensors import EXPORT_IGNORE_BLOCKS


class TestSharedIgnore:
    def test_mtp_fc_in_the_shared_list(self):
        assert "mtp.fc." in EXPORT_IGNORE_BLOCKS

    def test_tree_pin_synthesis_consults_the_list(self):
        from auto_round.compressors import orchestrator as orch_mod

        fn = getattr(orch_mod, "_resolve_tree_pins_", None)
        # the resolver may live on the orchestrator class; find its source either way
        if fn is None:
            for attr in dir(orch_mod):
                obj = getattr(orch_mod, attr)
                if inspect.isclass(obj) and hasattr(obj, "_resolve_tree_pins_"):
                    fn = getattr(obj, "_resolve_tree_pins_")
                    break
        assert fn is not None
        src = inspect.getsource(fn)
        assert "EXPORT_IGNORE_BLOCKS" in src
        assert '"bits"] = 16' in src.replace(" ", "") or 'entry["bits"] = 16' in src

    def test_ignore_substring_matches_module_names_like_tensors(self):
        # completer: "mtp.fc." in "mtp.fc.weight"; tree: "mtp.fc." in "mtp.fc."
        assert any(ign in "mtp.fc.weight" for ign in EXPORT_IGNORE_BLOCKS)
        assert any(ign in "mtp.fc." for ign in EXPORT_IGNORE_BLOCKS)
        # and ordinary tree linears are NOT ignored
        assert not any(ign in "mtp.layers.0.self_attn.q_proj." for ign in EXPORT_IGNORE_BLOCKS)
        assert not any(ign in "mtp.layers.0.mlp.down_proj." for ign in EXPORT_IGNORE_BLOCKS)
