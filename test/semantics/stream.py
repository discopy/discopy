from discopy import *
from discopy.stream import *


def test_feedback():
    T, S = Ty[python.Ty], Stream[python.Category]
    x, y, z = int, str, bool
    dom = T(x) @ T(y).delay()
    cod = T(z) @ T(y)
    now = python.Function(lambda n: (bool(n % 2), str(n)), (x, ), (z, y))
    later = python.Function(lambda n, s: (bool(n % 2), s + " " + str(n)), (x, y), (z, y))
    s = S(now, dom, cod, _later=lambda: S.constant(later))
    s.check_later()
    s.unroll().check_later()
    s.feedback(T(x), T(z), T(y)).unroll(2)
