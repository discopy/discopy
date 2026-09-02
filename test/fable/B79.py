"""B79: Bubble.subs and lambdify crash, a dict colour_map raises KeyError on empty types and PRO(0), substitute and tensor grad die on a foliated diagram (discopy/cat.py:559, monoidal.py:1633, 1644, 1283, tensor.py:645).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import sympy

from discopy import monoidal, tensor
from discopy.monoidal import Ty, Box, Id, Wire, Colour, Functor, PRO
from discopy.tensor import Dim

phi = sympy.Symbol('phi')
x, y, z = Ty('x'), Ty('y'), Ty('z')
f, g = Box('f', x, y), Box('g', y, z)
f_phi, f_one = Box('f', x, y, data=phi), Box('f', x, y, data=1)


def test_b79_bubble_subs():
    assert f_phi.bubble().subs(phi, 1) == f_one.bubble()


def test_b79_bubble_lambdify():
    assert f_phi.bubble().lambdify(phi)(1) == f_one.bubble()


red, green, pink, lime = map(Colour, ("red", "green", "pink", "lime"))
scalar = Box('s', Ty(), Ty())
coloured = Functor(
    {Ty(Wire("x", red, green)): Ty(Wire("X", pink, lime))}, {scalar: scalar},
    colour_map={red: pink, green: lime})


def test_b79_colour_map_on_empty_type():
    assert coloured(Ty()) == Ty()


def test_b79_colour_map_on_empty_identity():
    assert coloured(Id(Ty())) == Id(Ty())


def test_b79_colour_map_on_scalars():
    assert coloured(scalar >> scalar) == scalar >> scalar


class PRODiagram(monoidal.Diagram):
    ob = PRO


def test_b79_functor_on_pro_zero():
    F = Functor({}, {}, dom=PRODiagram, cod=monoidal.Diagram)
    assert F(PRO(0)) == Ty()


def test_b79_substitute_on_foliated_diagram():
    f2 = Box('f2', x, y)
    result = (f @ g).foliation().substitute(0, f2)
    assert result.boxes == [f2, g] and (result.dom, result.cod) == (x @ x, y @ z)


def test_b79_tensor_grad_on_foliated_diagram():
    box = tensor.Box('g', Dim(2), Dim(2), [phi, 0, 0, 1])
    expected = (box @ box).grad(phi).eval(dtype=object)
    result = (box @ box).foliation().grad(phi).eval(dtype=object)
    assert (result.array == expected.array).all()
