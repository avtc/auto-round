import enum
import importlib.util
import os
import sys
import types
from typing import Mapping

import pytest

# ---------------------------------------------------------------------------
# Environment workaround: torchvision 0.26 (installed here) is incompatible
# with torch 2.9 — ``import torchvision`` crashes with "operator
# torchvision::nms does not exist".  transformers imports torchvision at
# module import time in its optional vision stack, which breaks *any* model
# class import and therefore the whole test suite.  When (and only when)
# torchvision is broken, stub the handful of names transformers imports so
# the LLM-only tests can run.
# ---------------------------------------------------------------------------
_torchvision_spec = importlib.util.find_spec("torchvision")
try:
    import torchvision  # noqa: F401
except Exception:
    for _name in list(sys.modules):
        if _name == "torchvision" or _name.startswith("torchvision."):
            sys.modules.pop(_name, None)

    class _InterpolationMode(enum.IntEnum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 4
        BILINEAR = 2
        BICUBIC = 3
        HAMMING = 5
        LANCZOS = 6

    class _ImageReadMode(enum.IntEnum):
        UNCHANGED = 0
        GRAY = 1
        RGB = 2

    def _torchvision_unavailable(*args, **kwargs):
        raise RuntimeError("torchvision is broken in this environment (stubbed for tests)")

    _tv = types.ModuleType("torchvision")
    _tv.__spec__ = _torchvision_spec
    _tv_io = types.ModuleType("torchvision.io")
    _tv_io.ImageReadMode = _ImageReadMode
    _tv_io.decode_image = _torchvision_unavailable
    _tv_tr = types.ModuleType("torchvision.transforms")
    _tv_tr.InterpolationMode = _InterpolationMode
    _tv_fn = types.ModuleType("torchvision.transforms.functional")
    _tv_fn.pil_to_tensor = _torchvision_unavailable
    _tv_v2 = types.ModuleType("torchvision.transforms.v2")
    _tv_v2_fn = types.ModuleType("torchvision.transforms.v2.functional")

    def _tv_v2_fn_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _torchvision_unavailable

    _tv_v2_fn.__getattr__ = _tv_v2_fn_getattr
    _tv_v2.functional = _tv_v2_fn
    _tv.io = _tv_io
    _tv.transforms = _tv_tr
    _tv_tr.functional = _tv_fn
    _tv_tr.v2 = _tv_v2
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.io"] = _tv_io
    sys.modules["torchvision.transforms"] = _tv_tr
    sys.modules["torchvision.transforms.functional"] = _tv_fn
    sys.modules["torchvision.transforms.v2"] = _tv_v2
    sys.modules["torchvision.transforms.v2.functional"] = _tv_v2_fn

# Same issue with torchaudio: its C extension fails to load (ABI mismatch
# with torch 2.9) and transformers.loss.loss_rnnt imports it unconditionally.
_torchaudio_spec = importlib.util.find_spec("torchaudio")
try:
    import torchaudio  # noqa: F401
except Exception:
    for _name in list(sys.modules):
        if _name == "torchaudio" or _name.startswith("torchaudio."):
            sys.modules.pop(_name, None)

    def _torchaudio_unavailable(*args, **kwargs):
        raise RuntimeError("torchaudio is broken in this environment (stubbed for tests)")

    def _ta_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _torchaudio_unavailable

    _ta = types.ModuleType("torchaudio")
    _ta.__spec__ = _torchaudio_spec
    _ta.__file__ = None
    _ta.__getattr__ = _ta_getattr
    sys.modules["torchaudio"] = _ta
# ---------------------------------------------------------------------------

from .fixtures import *

# Easy debugging without installing auto-round.
sys.path.insert(0, "..")

# Workaround: some gguf builds report version 'N/A' which is not PEP 440
# compliant and causes packaging.version.InvalidVersion inside transformers.
try:
    import gguf as _gguf_mod
    from packaging.version import Version

    try:
        Version(_gguf_mod.__version__)
    except Exception:
        _gguf_mod.__version__ = "0.10.0"
except ImportError:
    pass


try:
    import torch

    # When loaded via the "meta" device, `gptqmodel==6.0.3` raises an error because the
    # internal loading process within the `transformers` library defaults to "meta" mode.
    # Importing under a CPU device context avoids that failure during module loading.
    with torch.device("cpu"):
        import gptqmodel  # pylint: disable=E0401
except ImportError:
    pass


### HPU related configuration, usage: `pytest --mode=compile/lazy``
def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="lazy",
        help="{compile|lazy}, default lazy. Choose mode to run tests",
    )


backup_env = pytest.StashKey[Mapping]()


def pytest_configure(config):
    pytest.mode = config.getoption("--mode")
    assert pytest.mode.lower() in ["lazy", "compile"]

    config.stash[backup_env] = os.environ

    if pytest.mode == "lazy":
        os.environ["PT_HPU_LAZY_MODE"] = "1"
    elif pytest.mode == "compile":
        os.environ["PT_HPU_LAZY_MODE"] = "0"
        os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"


def pytest_unconfigure(config):
    os.environ.clear()
    os.environ.update(config.stash[backup_env])
