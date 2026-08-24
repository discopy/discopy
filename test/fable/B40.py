"""B40: the --skip-extra contract is broken, torch/jax imports lack importorskip and pyzx has no version guard (test/tensor.py:15, test/quantum/zx.py).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from pathlib import Path

TEST = Path(__file__).parents[1]


def test_b40_torch_import_is_guarded():
    src = (TEST / 'tensor.py').read_text()
    assert ("importorskip('torch')" in src
            or 'importorskip("torch")' in src or "import torch" not in src)


def test_b40_jax_import_is_guarded():
    src = (TEST / 'tensor.py').read_text()
    assert ("importorskip('jax')" in src
            or 'importorskip("jax")' in src or "import jax" not in src)


def test_b40_pyzx_guard_is_version_aware():
    assert "minversion" in (TEST / 'quantum' / 'zx.py').read_text()
