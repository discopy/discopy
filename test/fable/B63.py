"""B63: rmap rebuilds an ndarray with the shape constructor, recurses forever on str data, and Box.subs and rsubs disagree on the shape of *args (discopy/utils.py:288, cat.py:549).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import sympy

from discopy.monoidal import Ty, Box

phi, psi = sympy.symbols("phi psi")
x, y = Ty('x'), Ty('y')


def test_b63_subs_on_ndarray_data():
    box = Box('f', x, x, data=np.array([phi, 1], dtype=object))
    result = box.subs(phi, 2).data
    assert result.shape == (2, ) and list(result) == [2, 1]


def test_b63_free_symbols_of_str_data():
    assert Box('f', x, y, data="hello").free_symbols == set()


def test_b63_subs_with_str_beside_symbol():
    box = Box('f', x, y, data=["hi", phi])
    assert box.subs(phi, 1).data == ["hi", 1]


def test_b63_subs_list_of_pairs():
    box = Box('f', x, y, data=phi + psi)
    assert box.subs([(phi, 1), (psi, 2)]).data == 3


def test_b63_subs_list_of_one_pair():
    box = Box('f', x, y, data=phi + psi)
    assert box.subs([(phi, 1)]).data == psi + 1


def test_b63_subs_pairs_as_args():
    box = Box('f', x, y, data=phi + psi)
    assert box.subs((phi, 1), (psi, 2)).data == 3
