from pytest import raises
from discopy.zx import *

def test_Spider():
    assert repr(Z(1, 2, .5)) == "Z(1, 2, .5)"
