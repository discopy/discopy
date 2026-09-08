# -*- coding: utf-8 -*-

"""
What a generator means and what a diagram compiles to.

A generator is read on the boundary of its box -- see
:class:`~discopy.neural.Network` -- and a closed diagram compiles into the
:class:`~discopy.neural.CMap` that runs it, with the ``(generator, role)``
:func:`~discopy.neural.families` of its ports.  The formulae of the module
docstring are pinned on a module that answers every port alike, so that
they hold for any width: one round is routing after interaction, injection
is an affine shift, iteration resumes bitwise.
"""

import os
import subprocess
import sys
from pathlib import Path

from pytest import importorskip, raises

from discopy.compact import Box, Cup, Ty as CompactTy
from discopy.frobenius import Box as Generator, Diagram, Ty
from discopy.neural import (
    Dim, Id, Network, Orbit, Para, Signature, Sym, families, heads, interpret)
from discopy.neural.map import width
from discopy.neural.signature import from_relation
from discopy.utils import AxiomError

torch = importorskip("torch")


PEER, STATE = Ty("peer"), Ty("state")

ROOT = Path(__file__).resolve().parents[2]


class Affine(torch.nn.Module):
    """ Answers every port alike, whatever the width of the box. """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.double))
        self.bias = torch.nn.Parameter(torch.tensor(1., dtype=torch.double))

    def forward(self, x):
        return self.weight * x + self.bias


def node_signature(degree: int = 1) -> Signature:
    return Signature((Orbit(PEER, degree, Sym.PERM),
                      Orbit(STATE, traced=True)))


OB = {PEER: Dim(3), STATE: Dim(4)}


def compiled(relation=((1, ), (0, 2), (1, ))):
    """ A path graph of sites sharing one module, in float64. """
    module = Affine()
    return module, interpret(
        from_relation(relation, node_signature(1)), OB, {"cell": module})


def test_compiling_a_diagram_does_not_import_torch():
    """
    Diagrams, signatures and the whole compilation layer load and run on a
    machine with no torch at all; only executing a module needs it.
    """
    script = (
        "import sys, discopy.neural as neural;"
        " from discopy.frobenius import Ty;"
        " peer = Ty('peer');"
        " node = neural.Signature((neural.Orbit(peer, 2), ));"
        " shape = neural.from_relation(((1, ), (0, )), node);"
        " found = neural.interpret(shape, {peer: neural.Dim(3)},"
        "                          {'cell': None});"
        " print(sum(found.port_widths), 'torch' in sys.modules)")
    found = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert found.stdout.strip() == "6 False"


def test_parametric_networks_compose_by_substitution():
    """
    An ordinary parametric map is a morphism of ``Para``: it composes, and
    the parameter objects go side by side, left then right.  ``Dim`` is a
    strict monoid, so the layout is associative on the nose and a
    bracketing can never change which weights are where.  A network in a
    diagram is not one of these: it is read on its boundary and talks
    along wires, see :class:`~discopy.neural.Network`.
    """
    f = Para.generator("f", Dim(2), Dim(3), Dim(6))
    g = Para.generator("g", Dim(3), Dim(4), Dim(12))
    h = Para.generator("h", Dim(4), Dim(5), Dim(20))
    assert (f.dom, f.cod, f.param) == (Dim(2), Dim(3), Dim(6))
    assert f.inside == Network("f", Dim(2, 6), Dim(3))
    assert (f >> g).param == Dim(6, 12) != g.param @ f.param
    assert (f >> g).inside.boxes[1] == Network("g", Dim(3, 12), Dim(4))
    assert ((f >> g) >> h).param == (f >> (g >> h)).param == Dim(6, 12, 20)
    assert ((f @ g) @ h).param == (f @ (g @ h)).param == Dim(6, 12, 20)
    assert ((f @ g) @ h).dom == (f @ (g @ h)).dom == Dim(2, 3, 4)
    shape = [(one.dom, one.cod, one.param) for one in (
        Para.id(Dim(2)) >> f, f, f >> Para.id(Dim(3)))]
    assert shape[0] == shape[1] == shape[2]
    with raises(AxiomError):
        f >> f


def test_heads_are_read_off_the_wiring():
    """
    A port is a head unless it is wired to an earlier port of the same box,
    which is exactly the second copy of a traced leg.  No declaration is
    consulted: the wiring says which ports a module reads a value off.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    ports, head_ports = families(pair, interpret(pair, OB, {"cell": None}), OB)
    assert ports["cell", STATE] == (1, 0, 4, 3)
    assert head_ports["cell", STATE] == (1, 4)
    assert ports["cell", PEER] == head_ports["cell", PEER] == (2, 5)
    assert heads(pair) == heads(pair.to_diagram()) == {
        ("cell", PEER): ((0, 0), (1, 0)), ("cell", STATE): ((0, 1), (1, 1))}


def test_heads_of_a_three_leg_orbit():
    """
    A traced orbit with three legs lays out its three outgoing copies before
    its three incoming ones, so the heads of the family are the first half
    of its ports in each box and every tail is the far end of its own head.
    """
    node = Signature((Orbit(PEER, 1), Orbit(STATE, 3, traced=True)))
    pair = from_relation(((1, ), (0, )), node)
    assert node.loops() == ((1, 4), (2, 5), (3, 6))
    assert heads(pair)["cell", STATE] == (
        (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3))
    cmap = interpret(pair, OB, {"cell": None})
    ports, head_ports = families(pair, cmap, OB)
    assert ports["cell", STATE] == (5, 4, 3, 2, 1, 0, 12, 11, 10, 9, 8, 7)
    assert head_ports["cell", STATE] == (5, 4, 3, 12, 11, 10)
    assert [cmap.edges[head] for head in head_ports["cell", STATE]]\
        == [2, 1, 0, 9, 8, 7]


def test_a_role_maps_to_an_atomic_dim():
    """
    One abstract port becomes one concrete port, or none: a role sent to a
    composite dimension has no port to become.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    with raises(ValueError, match="non-atomic"):
        interpret(pair, {PEER: Dim(3, 3), STATE: Dim(4)}, {"cell": None})


def test_only_a_closed_diagram_compiles():
    """
    A boundary port is one no box answers, so an open diagram has no global
    transition: ``interpret`` refuses a diagram and its map alike.
    """
    f = Network("f", Dim(2), Dim(3))
    for source in (f, f.to_map(), Id(Dim(2)).to_map()):
        with raises(ValueError, match="closed"):
            interpret(source, {}, {"f": None})


def test_erasing_a_role_erases_its_wires():
    """
    ``Dim(0)`` is the monoidal unit, so a role sent to it leaves neither a
    port nor a wire -- which is how one diagram serves two models.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    kept = interpret(pair, {PEER: Dim(3), STATE: Dim(5)}, {"cell": None})
    erased_ob = {PEER: Dim(3), STATE: Dim(0)}
    erased = interpret(pair, erased_ob, {"cell": None})
    assert kept.port_widths == (5, 5, 3, 5, 5, 3)
    assert erased.port_widths == (3, 3)
    assert ("cell", STATE) not in families(pair, erased, erased_ob)[1]
    assert width(pair, erased_ob) == 6 == sum(erased.port_widths)


def test_a_scalar_loop_survives_unless_erased():
    """
    A cap on a cup is a closed component with no ports: the compiled map
    keeps it, typed by the width of its role, so that a causal schedule
    refuses it as ``cmap`` does; a ``Dim(0)`` role erases it like a wire.
    """
    x = Ty("x")
    f, g = Generator("f", Ty(), x @ x), Generator("g", x @ x, Ty())
    source = (f >> g) @ (Diagram.caps(x, x) >> Diagram.cups(x, x))
    assert source.to_map().loops == (x, )
    kept = interpret(source, {x: Dim(2)}, {"f": None, "g": None})
    assert kept.loops == (Dim(2), ) and not kept.is_acyclic
    erased = interpret(source, {x: Dim(0)}, {"f": None, "g": None})
    assert erased.loops == () and erased.n_ports == 0 and erased.is_acyclic
    with raises(ValueError, match="non-atomic"):
        interpret(source, {x: Dim(2, 2)}, {"f": None, "g": None})


def test_a_dualised_role_reads_its_width_through_the_functor():
    """
    The functor sends a dual role to the dual of its image and an integer
    to an atomic dimension, so a compact source whose box carries ``x.r``
    compiles, and its families and its width are read off that functor.
    """
    x = CompactTy("x")
    closed = Box("f", CompactTy(), x @ x.r) >> Cup(x, x.r)
    cmap = interpret(closed, {x: Dim(2)}, {"f": None})
    assert cmap.port_widths == (2, 2) and tuple(cmap.edges) == (1, 0)
    ports, head_ports = families(closed, cmap, {x: 2})
    assert ports == {("f", x): (1, ), ("f", x.r): (0, )}
    assert head_ports == {("f", x): (1, )}
    assert width(closed, {x: 2}) == width(closed, {x: Dim(2)}) == 4


def test_sites_share_one_module():
    """
    Three sites of one name share one module, so the map has one module's
    worth of weights rather than the product of the sites' parameter
    objects.
    """
    module, cmap = compiled()
    assert [box.module for box in cmap.boxes] == [module] * 3
    wrapped = cmap.as_network().module
    assert sum(p.numel() for p in wrapped.parameters()) \
        == sum(p.numel() for p in module.parameters()) == 2


def test_a_map_keeps_the_shape_it_was_built_with():
    """
    :attr:`CMap.port_widths` and :attr:`CMap.routing` are cached, which is
    sound exactly as long as a map's boxes are what its constructor fixed:
    the boxes, the ports, the wiring and the widths recomputed from scratch
    are identical after a map has been run.
    """
    _, cmap = compiled()

    def shape():
        return (tuple(sum(getattr(port.obj, "inside", (port.obj, )))
                      for port in cmap.ports),
                tuple(cmap.boxes), tuple(cmap.ports), tuple(cmap.edges))

    before = shape()
    assert cmap.port_widths == before[0]
    state = torch.rand(2, sum(cmap.port_widths)).double()
    with torch.no_grad():
        for _ in range(3):
            state = cmap(init=state, n_rounds=1, return_flat=True)
    assert shape() == before and cmap.port_widths == shape()[0]


def test_read_and_write_address_a_family():
    pair = from_relation(((1, ), (0, )), node_signature(1))
    cmap = interpret(pair, OB, {"cell": Affine()})
    ports, head_ports = families(pair, cmap, OB)
    state = cmap.zeros(2, like=torch.zeros(1, dtype=torch.double))
    values = torch.arange(2 * 4 * 4, dtype=torch.double).reshape(2, 4, 4)
    written = cmap.write(state, ports["cell", STATE], values)
    assert torch.equal(cmap.read(written, ports["cell", STATE]), values)
    assert torch.equal(
        cmap.read(written, head_ports["cell", STATE]), values[:, 0::2])
    assert cmap.read(written, head_ports["cell", PEER]).abs().sum() == 0
    assert state.shape == (2, sum(cmap.port_widths))
    with raises(ValueError, match="different widths"):
        cmap.read(state, (0, 2))


def test_round_is_routing_after_interaction():
    """
    ``T(s) = sigma(Phi(s))``: one round with no reinjection is the boxes'
    emissions carried along the wires.
    """
    _, cmap = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)
    with torch.no_grad():
        emitted = cmap(init=state, n_rounds=1, inject=False)
        routed = cmap(init=state, n_rounds=1, inject=False, return_flat=True)
    widths = cmap.port_widths
    per_port = [None] * len(widths)
    for index, chunks in enumerate(emitted):
        ports = cmap.box_ports(index)
        for port, chunk in zip(ports, torch.split(
                chunks, [widths[port] for port in ports], -1)):
            per_port[port] = chunk
    expected = torch.cat(
        [per_port[cmap.edges[port]] for port in range(len(widths))], -1)
    assert torch.equal(expected, routed)


def test_reinjection_is_an_affine_shift():
    """ ``T(s) = sigma(Phi(s)) + i``: the initial vector is added back. """
    _, cmap = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)
    with torch.no_grad():
        plain = cmap(init=state, n_rounds=1, inject=False, return_flat=True)
        injected = cmap(init=state, n_rounds=1, inject=True, return_flat=True)
    assert torch.equal(injected, plain + state)


def test_iteration_is_resumption():
    """
    ``T^(a+b) = T^b . T^a``, bitwise, and it holds for *one* transition, so
    a run resumed from its own carried state only resumes when ``inject``
    is off.
    """
    _, cmap = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)

    def advance(state, rounds, inject=False):
        return cmap(init=state, n_rounds=rounds, inject=inject,
                    return_flat=True)

    with torch.no_grad():
        whole = advance(state, 5)
        resumed = advance(advance(state, 2), 3)
        injected = advance(state, 5, inject=True)
        piecewise = advance(advance(state, 2, inject=True), 3, inject=True)
        rounds = cmap(init=state, n_rounds=5, inject=False,
                      return_rounds=True, return_flat=True)
    assert torch.equal(whole, resumed)
    assert torch.equal(rounds[1], advance(state, 2))
    assert not torch.equal(injected, piecewise)
    assert len(rounds) == 5 and torch.equal(rounds[-1], whole)


def test_a_product_of_diagrams_is_a_product_of_states():
    """
    A batch of instances is the monoidal product of their maps, and the
    state of the product is the sum of the states: one summand per port,
    the members' ports in order.
    """
    _, pair = compiled(((1, ), (0, )))
    _, path = compiled(((1, ), (0, 2), (1, )))
    node = node_signature(1)
    both = interpret(
        from_relation(((1, ), (0, )), node)
        @ from_relation(((1, ), (0, 2), (1, )), node), OB, {"cell": Affine()})
    assert both.port_widths == pair.port_widths + path.port_widths
    assert len(both.boxes) == 5


def test_a_snake_is_pure_rerouting():
    """
    Swaps, cups and caps are wiring, which a functor preserves strictly and
    for free: a snake has no box at all, and its forward pass is the
    identity.
    """
    snake = Id(Dim(2)).transpose().to_map()
    assert snake.boxes == () and snake.port_widths == (2, 2)
    x = torch.tensor([[0.1, 0.2]])
    assert torch.equal(snake(x), x)
