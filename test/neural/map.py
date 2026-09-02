# -*- coding: utf-8 -*-

"""
What a generator means and what a diagram compiles to.

The two readings of a generator -- a :class:`~discopy.neural.ParamMap` that
composes by substitution, an :class:`~discopy.neural.InteractionMap` that
does not -- and the compilation of a closed diagram into the
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

import pytest

torch = pytest.importorskip("torch")

from discopy.frobenius import Ty
from discopy.para import Symmetric
from discopy.neural import (
    Dim, Network, Orbit, Signature, Sym, families, interpret)
from discopy.neural.map import InteractionMap, ParamMap, interaction_spec
from discopy.neural.signature import from_relation
from discopy.utils import AxiomError


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


def test_param_maps_compose_by_substitution():
    """
    An ordinary parametric map is a morphism of ``Para``: it composes, and
    the parameter objects go side by side, left then right.  ``Dim`` is a
    strict monoid, so the layout is associative on the nose and a
    bracketing can never change which weights are where.
    """
    f = ParamMap.generator("f", Dim(2), Dim(3), Dim(6))
    g = ParamMap.generator("g", Dim(3), Dim(4), Dim(12))
    h = ParamMap.generator("h", Dim(4), Dim(5), Dim(20))
    assert (f.name, f.dom, f.cod, f.params) == ("f", Dim(2), Dim(3), Dim(6))
    assert isinstance(f, Symmetric) and ParamMap.id(Dim(2)).params == Dim()
    assert (f >> g).params == Dim(6, 12) != (g.params @ f.params)
    assert ((f >> g) >> h).params == (f >> (g >> h)).params == Dim(6, 12, 20)
    assert ((f @ g) @ h).params == (f @ (g @ h)).params == Dim(6, 12, 20)
    assert ((f @ g) @ h).dom == (f @ (g @ h)).dom == Dim(2, 3, 4)
    with pytest.raises(AxiomError, match="does not compose"):
        f >> f


def test_interaction_maps_do_not_compose():
    """
    The refusal that keeps the two readings apart.  Two interactions glued
    along a shared object talk along the wires -- symmetric feedback, i.e.
    the trace of the two boxes over the shared boundary -- and what
    computes it is a finite number of rounds, not a substitution.  Their
    *tensor* is meaningful and is kept.
    """
    f = InteractionMap.generator("f", Dim(2, 3), Dim(4, 5, 6))
    g = InteractionMap.generator("g", Dim(7), Dim(8))
    with pytest.raises(AxiomError, match="do not compose"):
        f >> g
    with pytest.raises(TypeError):
        ParamMap.generator("f", Dim(2), Dim(3)) @ g
    assert ParamMap.generator("f", Dim(2), Dim(3)) \
        != InteractionMap.generator("f", Dim(2), Dim(3))
    assert f.boundary == Dim(2, 3, 4, 5, 6) and f.width == 20
    assert f.dagger().boundary == Dim(4, 5, 6, 2, 3)
    assert f.dagger().width == 20 and f.dagger().dagger() == f
    assert (f @ g).boundary == Dim(2, 3, 7, 4, 5, 6, 8)
    assert (f @ g).boundary != f.boundary @ g.boundary
    assert (f @ g).width == f.width + g.width


def test_interaction_spec_reads_a_network():
    torch.manual_seed(0)
    module = torch.nn.Linear(5, 5)
    f = Network("f", Dim(2), Dim(3), module=module)
    spec = interaction_spec(f)
    assert spec == InteractionMap.generator("f", Dim(2), Dim(3), Dim(30))
    assert spec.boundary == f.dom @ f.cod
    assert spec.width == module.in_features == module.out_features
    assert interaction_spec(f.dagger()) == spec.dagger()
    assert interaction_spec(Network("g", Dim(2), Dim(3))).params == Dim()
    assert interaction_spec(
        Network("g", Dim(2), Dim(3), module=object())).params == Dim()


def test_the_specs_agree_with_the_map():
    module, cmap = compiled()
    specs = [interaction_spec(box) for box in cmap.boxes]
    assert [spec.name for spec in specs] == ["cell", "cell", "cell"]
    for index, (spec, box) in enumerate(zip(specs, cmap.boxes)):
        assert spec.boundary == box.dom @ box.cod
        assert spec.width == sum(
            cmap.port_widths[port] for port in cmap.box_ports(index))
    assert {spec.params for spec in specs} == {
        Dim(sum(p.numel() for p in module.parameters()))}
    # sites share one module, so theta is *not* the product of their
    # parameter objects: the map has one module's worth of weights.
    assert sum(p.numel() for p in cmap.parameters()) \
        == sum(p.numel() for p in module.parameters())


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


def test_heads_are_read_off_the_wiring():
    """
    A port is a head unless it is wired to an earlier port of the same box,
    which is exactly the second copy of a traced leg.  No declaration is
    consulted: the wiring says which ports a module reads a value off.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    ports, heads = families(pair, interpret(pair, OB, {"cell": None}), OB)
    assert ports["cell", STATE] == (1, 0, 4, 3)
    assert heads["cell", STATE] == (1, 4)
    assert ports["cell", PEER] == heads["cell", PEER] == (2, 5)


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
    with pytest.raises(ValueError, match="non-atomic"):
        interpret(pair, {PEER: Dim(3, 3), STATE: Dim(4)}, {"cell": None})
    with pytest.raises(ValueError, match="closed"):
        interpret(Network("f", Dim(2), Dim(3)).to_map(), {}, {"f": None})


def test_read_and_write_address_a_family():
    pair = from_relation(((1, ), (0, )), node_signature(1))
    cmap = interpret(pair, OB, {"cell": Affine()})
    ports, heads = families(pair, cmap, OB)
    state = cmap.zeros(2, like=torch.zeros(1, dtype=torch.double))
    values = torch.arange(2 * 4 * 4, dtype=torch.double).reshape(2, 4, 4)
    written = cmap.write(state, ports["cell", STATE], values)
    assert torch.equal(cmap.read(written, ports["cell", STATE]), values)
    assert torch.equal(
        cmap.read(written, heads["cell", STATE]), values[:, 0::2])
    assert cmap.read(written, heads["cell", PEER]).abs().sum() == 0
    assert state.shape == (2, sum(cmap.port_widths))
    with pytest.raises(ValueError, match="different widths"):
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
    from discopy.neural import Id
    snake = Id(Dim(2)).transpose().to_map()
    assert snake.boxes == () and snake.port_widths == (2, 2)
    x = torch.tensor([[0.1, 0.2]])
    assert torch.equal(snake(x), x)
