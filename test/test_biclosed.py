from pytest import raises
from discopy.biclosed import *


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert repr(Over(x, y)) == "Over(Ty('x'), Ty('y'))"

def test_Under():
    x, y = Ty('x'), Ty('y')
    assert repr(Under(x, y)) == "Under(Ty('x'), Ty('y'))"
