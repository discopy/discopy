"""B45: doc and message nits — self-contradictory or type-only error messages, a stale plugin pointer and a missing CY (discopy/python/finset.py:127, biclosed.py:671, __init__.py, quantum).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from pathlib import Path

import pytest

import discopy.quantum
from discopy import biclosed
from discopy.python.finset import Permutation

ROOT = Path(__file__).parents[2]


def test_b45_permutation_message_is_not_self_contradictory():
    with pytest.raises(ValueError) as exc:
        Permutation((0, 0))
    assert "2, got 2" not in str(exc.value)


def test_b45_application_message_names_the_value():
    v = biclosed.Variable('v', biclosed.Ty('x'))
    w = biclosed.Variable('w', biclosed.Ty('y'))
    with pytest.raises(TypeError) as exc:
        biclosed.Application(v, w)
    assert "<class" not in str(exc.value)


def test_b45_no_stale_pytest_plugin_pointer():
    src = (ROOT / 'discopy' / '__init__.py').read_text()
    assert "pytest_plugin" not in src


def test_b45_quantum_has_cy():
    assert hasattr(discopy.quantum, 'CY')
