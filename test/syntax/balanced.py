from discopy.balanced import *


def test_repr():
    x = Ty('x')
    assert repr(Twist(Ty('x')))\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x')))"
    assert repr(Twist(Ty('x')).dagger())\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x'))).dagger()"
