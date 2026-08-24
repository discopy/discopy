"""B44: quantum interop tests only exercise phase 0 and never evaluate numerically (test/quantum/zx.py:159, test/quantum/tk.py).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import re
from pathlib import Path

TEST = Path(__file__).parents[1]


def test_b44_zx_translates_a_nonzero_phase_controlled_gate():
    src = (TEST / 'quantum' / 'zx.py').read_text()
    assert re.search(r"CR[xz]\(0\.[0-9]*[1-9]", src)


def test_b44_tk_roundtrip_is_numerical():
    assert ".eval()" in (TEST / 'quantum' / 'tk.py').read_text()
