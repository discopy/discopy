# -*- coding: utf-8 -*-

import subprocess
import sys

from pytest import raises

from discopy import neural, python
from discopy.neural.network import (
    Box, Copy, Dim, Dims, Discard, Equation, Functor, Id, Merge, Network,
    Permutation, Swap, Trace)
from discopy.utils import AxiomError, dumps, loads


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.neural; "
        "assert 'torch' not in sys.modules"], check=True)


def test_dims():
    assert Dims(2, 3) == Dims(Dim(2), Dim(3)) == Dims(2) @ Dims(3)
    assert Dims(2, 3) + Dims(4) == Dims(2, 3, 4) and Dims(2) ** 2 == Dims(2, 2)
    assert Dims() != Dims(1) == Dims(Dim()) and Dims(Dim(2, 3)) != Dims(2, 3)
    assert Dims(2, 3) * Dims(4, 5) == Dims(Dim(2, 4), Dim(2, 5), Dim(3, 4),
                                           Dim(3, 5))
    assert Dims(2) * Dims() == Dims() and Dims(2) * Dims(1) == Dims(2)
    assert Dims(2, 3)[1:] == Dims(3) and list(Dims(2, 3)) == [Dims(2), Dims(3)]
    assert repr(Dims(2, Dim(2, 3))) == str(Dims(2, Dim(2, 3)))\
        == "Dims(Dim(2), Dim(2, 3))"
    assert loads(dumps(Dims(2, Dim(2, 3)))) == Dims(2, Dim(2, 3))
    assert hash(Dims(2)) == hash(Dims(Dim(2)))
    with raises(TypeError):
        Dims('x')
    with raises(ValueError):
        Dims(0)
    with raises(TypeError):
        Dims(2) * Dim(2)


def test_box():
    f = Box('f', Dims(2), Dims(3, 3), module=lambda v: (v, v))
    assert f.module is f.data is f.dagger().module and f.module(1) == (1, 1)
    assert repr(f) == "neural.network.Box('f', Dims(Dim(2)), "\
        "Dims(Dim(3), Dim(3)))"
    assert repr(f.dagger()) == repr(f) + ".dagger()"
    assert Box('f', Dims(2), Dims(3, 3)) != f
    assert Box('f', Dims(2), Dims(3, 3), data=f.module) == f
    assert loads(dumps(f)) == Box('f', Dims(2), Dims(3, 3))
    one, other = (Box('f', Dims(2), Dims(2), module=object()) for _ in "12")
    assert one != other and one == one.dagger().dagger()
    scope = {"neural": neural, "Dims": Dims, "Dim": Dim}
    for box in (Box('f', Dims(2), Dims(3)), Box('f', Dims(2), Dims(3))[::-1]):
        assert eval(repr(box), scope) == box


def test_structure():
    x, y = Dims(2), Dims(3)
    assert Network.swap(x, y) == Swap(x, y) and Swap(x, y).dagger()\
        == Swap(y, x)
    assert Network.swap(Dims(), x) == Id(x)
    assert Network.from_permutation([2, 0, 1], x @ y @ x)\
        == Permutation(x @ y @ x, [2, 0, 1])
    assert Network.copy(x) == Copy(x) and Network.copy(x, 0) == Discard(x)
    assert Network.copy(x @ y) == Copy(x) @ Copy(y) >> x @ Swap(x, y) @ y
    assert Network.merge(x) == Merge(x) == Copy(x).dagger()
    assert Merge(x).dagger() == Copy(x) and Discard(x).dagger() == Merge(x, 0)
    assert isinstance(Network.discard(x).boxes[0], Discard)
    assert isinstance(Network.merge(x).boxes[0], Merge)
    for box in (Copy(x), Merge(x, 3), Discard(x)):
        assert loads(dumps(box)) == box
    with raises(ValueError):
        Copy(x @ y)


def test_trace():
    x, h = Dims(2), Dims(4)
    cell = Box('cell', x @ h, x @ h)
    assert isinstance(cell.trace(), Trace)
    assert cell.trace().dom == cell.trace().cod == x
    assert cell.trace(left=True).dom == h and cell.trace(0) == cell
    assert cell.trace().dagger() == cell.dagger().trace()
    assert loads(dumps(cell.trace())) == cell.trace()
    with raises(AxiomError):
        Box('f', x, h).trace()
    assert not cell.trace().to_map().is_acyclic


def test_from_callable():
    x = Dims(4)
    layer, add = Box('layer', x, x), Box('add', x @ x, x)

    @Network.from_callable(x, x)
    def residual(v):
        return add(layer(v), v)

    assert residual == Network.copy(x) >> layer @ x >> add
    assert loads(dumps(residual)) == residual
    assert residual.to_hypergraph().boxes == (layer, add)


def test_equation():
    x = Dims(2)
    f, g = Box('f', x @ x, x @ x), Box('g', x, x)
    assert Equation((f >> x @ g).trace(), (x @ g >> f).trace())
    assert not Equation(f.trace(), f.trace(left=True))
    assert Equation(Swap(x, x).trace(), Id(x))


def test_functor():
    x = Dims(2)
    layer, add = Box('layer', x, x), Box('add', x @ x, x)
    residual = Network.copy(x) >> layer @ x >> add
    F = Functor(
        ob_map={x: (list, )},
        ar_map={layer: lambda v: [2 * i for i in v],
                add: lambda v, w: [i + j for i, j in zip(v, w)]},
        cod=python.Function)
    assert F(residual)([1, 2]) == [3, 6]
    assert F(x @ x) == (list, list) and F(Dims()) == ()
    G = Functor(ob_map={x: x @ x},
                ar_map={layer: add >> add.dagger(), add: add @ add})
    assert G(residual)\
        == Network.copy(x @ x) >> (add >> add.dagger()) @ x @ x >> add @ add
    assert G(residual.trace()) == G(residual).trace(2)
