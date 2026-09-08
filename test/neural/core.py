# -*- coding: utf-8 -*-

import subprocess
import sys

from pytest import importorskip, raises

from discopy import neural
from discopy.neural import (
    CMap, Cap, Cup, Diagram, Dim, Equation, Functor, Id, Network, Para, Swap)
from discopy.utils import AxiomError, dumps, loads


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.neural; "
        "assert 'torch' not in sys.modules"], check=True)


def test_dim():
    assert Dim(0) == Dim() == Dim(0, 0)
    assert Dim(0) @ Dim(2) == Dim(2) and Dim(2) @ Dim(3) == Dim(2, 3)
    assert Dim(2).l == Dim(2).r == Dim(2)
    assert Dim(2, 3).r == Dim(3, 2)
    assert loads(dumps(Dim(2, 3))) == Dim(2, 3)
    with raises(ValueError):
        Dim(-1)


def test_axioms():
    x = Dim(2)
    assert Equation(
        Id(x).transpose(), Id(x), Id(x).transpose(left=True))
    assert Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
    assert Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))


def test_network_as_box():
    f = Network('f', Dim(2), Dim(3))
    g = Network('g', Dim(3), Dim(2))
    assert (f >> g).dom == Dim(2) and (f @ g).cod == Dim(3, 2)
    assert f.dagger().dom == Dim(3) and f.rotate().cod == Dim(2)
    assert repr(f) == "neural.core.Network('f', Dim(2), Dim(3))"
    assert Network('f', Dim(2), Dim(3)) == Network('f', Dim(2), Dim(3))
    one, other = (
        Network('f', Dim(2), Dim(3), module=object()) for _ in range(2))
    assert one != other and one == one.dagger().dagger()
    with raises(TypeError):
        hash(Network('f', Dim(2), Dim(3), module=[]))
    stateful = Network('f', Dim(2), Dim(3), mem=Dim(4))
    assert stateful != f
    assert stateful.dagger().mem == stateful.rotate().mem == Dim(4)
    assert stateful == stateful.dagger().dagger()
    assert stateful.to_map().port_widths == f.to_map().port_widths


def test_network_repr():
    f = Network('f', Dim(2), Dim(3), mem=Dim(1))
    scope = {"neural": neural, "Dim": Dim}
    for network in (f, f.dagger(), f.rotate(), f.rotate().dagger()):
        assert eval(repr(network), scope) == network
    assert f.rotate().z == 1 and f.rotate().rotate() == f
    assert f.dagger().is_dagger and f.dagger().dagger() == f


def test_network_serialisation():
    f = Network('f', Dim(2), Dim(3), mem=Dim(4))
    for network in (f, f.rotate(), f.dagger()):
        assert Network.from_tree(network.to_tree()) == network
        assert loads(dumps(network)) == network
    assert f.rotate().to_tree()["z"] == 1 and "z" not in f.to_tree()
    with_module = Network('g', Dim(2), Dim(3), module=object(), mem=Dim(1))
    assert loads(dumps(with_module))\
        == Network('g', Dim(2), Dim(3), mem=Dim(1))


def test_network_call():
    double = Network('double', Dim(1), Dim(1), module=lambda value: 2 * value)
    assert double(3) == 6 and double.module is double.data


def test_port_widths():
    f = Network('f', Dim(2, 3), Dim(4, 5, 6))
    fm = f.to_map()
    assert fm.port_widths == (2, 3, 2, 3, 6, 5, 4, 4, 5, 6)


def test_to_hypergraph():
    f = Network('f', Dim(2), Dim(3), mem=Dim(4))
    hypergraph = f.to_hypergraph()
    round_trip = hypergraph.to_diagram()
    assert round_trip.to_hypergraph() == hypergraph
    assert tuple(round_trip.boxes) == (f, )


def test_functor():
    f = Network('f', Dim(2), Dim(3), mem=Dim(1))
    image = Network('F(f)', Dim(4), Dim(3), mem=Dim(1))
    functor = Functor(
        ob_map={Dim(2): Dim(4), Dim(3): Dim(3)}, ar_map={f: image})
    assert functor(Dim(2, 3)) == Dim(4, 3) and functor(Dim()) == Dim()
    assert functor(f) == image and functor(f).mem == Dim(1)
    assert functor(f >> f.dagger()) == image >> image.dagger()
    assert functor(f.rotate()) == image.rotate()
    assert functor(Diagram.swap(Dim(2), Dim(3)))\
        == Diagram.swap(Dim(4), Dim(3))
    assert functor(Diagram.cups(Dim(2), Dim(2)))\
        == Diagram.cups(Dim(4), Dim(4))


def test_from_wiring_errors():
    f = Network('f', Dim(1), Dim(1), module=object())
    with raises(ValueError, match="has no port"):
        CMap.from_wiring((f, ), [((0, 0), (0, 2))])
    with raises(ValueError, match="wired to itself"):
        CMap.from_wiring((f, ), [((0, 0), (0, 0))])
    with raises(ValueError, match="wired twice"):
        CMap.from_wiring((f, f), [((0, 0), (0, 1)), ((0, 0), (1, 1))])
    with raises(ValueError, match="left unwired"):
        CMap.from_wiring((f, f), [((0, 0), (0, 1))])
    closed = CMap.from_wiring((f, f), [((0, 0), (1, 1)), ((0, 1), (1, 0))])
    assert closed.n_ports == 4 and closed.is_monogamous


def test_para():
    torch = importorskip("torch")

    class Multiply(torch.nn.Module):
        """ Read an input and a weight, emit their product. """
        def forward(self, messages):
            value = messages[:, :2]
            product = value[:, :1] * value[:, 1:]
            return torch.cat((torch.zeros_like(value), product), dim=-1)

    scale = Para(Dim(1), Dim(1), Network(
        "scale", Dim(1, 1), Dim(1), module=Multiply()), Dim(1))
    network = scale >> scale
    assert (network.dom, network.cod, network.param)\
        == (Dim(1), Dim(1), Dim(1, 1))
    assert network.inside.dom == Dim(1, 1, 1)
    assert (scale @ scale).param == Dim(1, 1)
    with raises(AxiomError):
        Para(Dim(1), Dim(1), scale.inside, Dim(2))


def test_para_generator():
    linear = Para.generator("linear", Dim(2), Dim(2), Dim(4))
    assert linear == Para(
        Dim(2), Dim(2), Network("linear", Dim(2, 4), Dim(2)), Dim(4))
    plain = Para.generator("f", Dim(1), Dim(1))
    assert plain.param == Dim()
    assert plain.inside == Network("f", Dim(1), Dim(1))


def test_network_module():
    torch = importorskip("torch")
    module = torch.nn.Linear(5, 5)
    f = Network('f', Dim(2), Dim(3), module=module)
    assert f.module(torch.ones(4, 5)).shape == (4, 5)
    assert f(torch.ones(1, 5)).shape == (1, 5)
    assert f.module is f.data is f.dagger().module
