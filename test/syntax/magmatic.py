from __future__ import annotations

from pytest import raises

from discopy import monoidal
from discopy.magmatic import *
from discopy.magmatic import Tensor
from discopy.utils import dumps, loads


def test_Ty():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert x & y == x.pack(y) == Ty(Tensor(x, y))
    assert (x & y).is_tensor and not (x @ y).is_tensor
    assert ((x & y) & z).left == x & y and ((x & y) & z).right == z
    assert (x & (y & z)).unpack() == x @ (y & z)
    assert ((x & y) & z).flatten() == x @ y @ z == (x @ y @ z).flatten()
    with raises(AssertionError):
        (x @ y).left


def test_Tensor():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        Tensor('x', y)
    assert Tensor(x, y) == Tensor(x, y) != Tensor(y, x)
    assert {Tensor(x, y): 42}[Tensor(x, y)] == 42
    assert str(Tensor(x, y)) == "(x & y)"
    assert repr(Tensor(x, y)) == "magmatic.Tensor("\
        "magmatic.Ty(cat.Ob('x')), magmatic.Ty(cat.Ob('y')))"
    assert loads(dumps(x & (y & x))) == x & (y & x)


def test_Pack():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        Pack(x, 'y')
    with raises(TypeError):
        Unpack('x', y)
    assert Pack(x, y).dom == x @ y and Pack(x, y).cod == x & y
    assert Unpack(x, y).dom == x & y and Unpack(x, y).cod == x @ y
    assert Diagram.pack(x, y) == Pack(x, y)
    assert Diagram.unpack(x, y) == Unpack(x, y)
    assert Pack(x, y).dagger() == Unpack(x, y)
    assert Unpack(x, y).dagger() == Pack(x, y)
    assert repr(Pack(x, y))\
        == "magmatic.Pack(magmatic.Ty(cat.Ob('x')), magmatic.Ty(cat.Ob('y')))"
    assert repr(Unpack(x, y)) == repr(Pack(x, y)).replace("Pack", "Unpack")
    assert loads(dumps(Pack(x, y) >> Unpack(x, y)))\
        == Pack(x, y) >> Unpack(x, y)


def test_Pair():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    assert f & g == Pair(f, g)
    assert f & y == f & Diagram.id(y)  # types are whiskered to identities
    assert (f & g).dom == x & y and (f & g).cod == y & x
    assert (f & g).left == f and (f & g).right == g
    assert (f & g).decompose() == Unpack(x, y) >> f @ g >> Pack(y, x)
    assert (f & g).dagger() == f.dagger() & g.dagger()
    assert str(f & g) == "(f & g)"
    assert loads(dumps(f & g)) == f & g


def test_Functor():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, x)
    F = Functor({x: y, y: z, z: x}, {f: Box('f', y, z), g: Box('g', z, y)})
    assert F(x & (y & z)) == y & (z & x)
    assert F(Pack(x, y)) == Pack(y, z)
    assert F(Unpack(x, y)) == Unpack(y, z)
    assert F(f & g) == Box('f', y, z) & Box('g', z, y)


def test_to_monoidal():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    f_, g_ = f.to_monoidal(), g.to_monoidal()
    assert f_ == monoidal.Box('f', monoidal.Ty('x'), monoidal.Ty('y'))
    assert (f & g).to_monoidal() == f_ @ g_
    assert (Pack(x, y) >> Unpack(x, y) >> f @ g).to_monoidal() == f_ @ g_
    assert Diagram.from_monoidal(f_ @ g_) == f @ g
    assert (f & g).decompose().to_monoidal() == f_ @ g_


def test_drawing():
    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    (Pack(x, y) >> (f & g).decompose()).to_drawing()
    (f & g).to_drawing()
