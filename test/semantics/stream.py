from discopy import *
from discopy.stream import *



T, S = Ty[python.Ty], Stream[python.Category]
x, y, m = int, bool, str
dom = T(x) @ T(m).delay()
cod = T(y) @ T(m)
now = python.Function(lambda n: (bool(n % 2), str(n)), (x, ), (y, m))
later = python.Function(lambda n, s: (bool(n % 2), s + " " + str(n)), (x, m), (y, m))
s = S(now, dom, cod, _later=lambda: S.constant(later))
s.check_later()
s.unroll().check_later()
s.feedback(T(x), T(y), T(m)).unroll(2)
