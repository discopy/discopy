"""B34: nesting never checks the two types have equal length (discopy/rigid.py:867).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import pytest

from discopy import rigid
from discopy.utils import AxiomError


def test_b34_cups_empty_left_raises():
    with pytest.raises(AxiomError):
        rigid.Diagram.cups(rigid.Ty(), rigid.Ty('n'))


def test_b34_cups_empty_right_raises():
    with pytest.raises(AxiomError):
        rigid.Diagram.cups(rigid.Ty('n'), rigid.Ty())
