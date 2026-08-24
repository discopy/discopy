# -*- coding: utf-8 -*-
"""B24: pennylane.py calls Operation.inv(), removed in PennyLane 0.32, and scales probabilities by _scale**2 instead of abs(_scale)**2 (discopy/quantum/pennylane.py:68).
Asserts the correct behaviour, red while the bug is live — issue #606.
This is a static source check: torch is not installed, so the module is
read as text with pathlib rather than imported.
"""
from pathlib import Path

import discopy

SOURCE = Path(discopy.__file__).parent / 'quantum' / 'pennylane.py'


def test_b24_no_removed_inv():
    assert '.inv()' not in SOURCE.read_text()


def test_b24_scale_uses_abs():
    assert 'abs(self._scale)' in SOURCE.read_text()
