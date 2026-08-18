# -*- coding: utf-8 -*-

"""
The semantics: what the executable code *means*, checked against the
executable code.

Nothing here is a second numerical backend, so nothing here asserts a
tensor value on its own.  What it asserts is agreement: that
:func:`~discopy.neural.map.interaction_spec` reproduces the boundary of a
:class:`~discopy.neural.Network` without touching it, that a compiled
:class:`~discopy.neural.map.Interaction` reproduces the state width and the
routing of the :class:`~discopy.neural.CMap` it runs, that the formulae the
docstrings state -- ``T(s) = sigma(Phi(s)) + i`` and ``T^(a+b) = T^b . T^a``
-- are the ones the forward pass computes, and that a law of
:mod:`discopy.neural.laws` is the symmetry
:func:`~discopy.neural.laws.check_equivariant` measures.

It also pins the two refusals.  An :class:`~discopy.neural.InteractionMap`
does not compose by substitution -- gluing two of them is wiring plus
iteration -- and a diagram with a boundary does not compile to an
interaction, because a transition of the state alone needs every port to
belong to a box.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from discopy.frobenius import Ty
from discopy.neural import (
    Dim, Iterate, Mode, Network, Orbit, Relation, Signature, Site, Sym, laws)
from discopy.neural.laws import check_equivariant, fusion_residual
from discopy.neural.map import (
    Interaction, InteractionMap, ParamMap, Parametric, interaction_spec,
    interpret, route)
from discopy.neural.signature import from_relation
from discopy.utils import AxiomError

PEER, STATE, MESSAGE = Ty("peer"), Ty("state"), Ty("message")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def node_signature(degree: int = 1) -> Signature:
    return Signature((Orbit(PEER, degree, Sym.PERM),
                      Orbit(STATE, traced=True)))


def compiled(relation=((1, ), (0, 2), (1, ))):
    """ A path graph of sites sharing one module, in float64. """
    node = node_signature(1)
    torch.manual_seed(0)
    module = Site(node, {PEER: 3, STATE: 4}, {STATE: Mode.STATE},
                  hidden=6).double()
    return module, interpret(from_relation(relation, node),
                             {PEER: Dim(3), STATE: Dim(4)},
                             {"cell": module})


# --- the compilation layer is torch-free -----------------------------------

def test_compiling_a_diagram_does_not_import_torch():
    """
    Diagrams, signatures, laws and the whole compilation layer load and run
    on a machine with no torch at all; only executing a module needs it.
    """
    script = (
        "import sys, discopy.neural as neural;"
        " from discopy.frobenius import Ty;"
        " peer = Ty('peer');"
        " node = neural.Signature((neural.Orbit(peer, 2), ));"
        " shape = neural.from_relation(((1, ), (0, )), node);"
        " found = neural.interpret(shape, {peer: neural.Dim(3)},"
        "                          {'cell': None});"
        " print(found.total, 'torch' in sys.modules)")
    found = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert found.stdout.strip() == "6 False"


# --- what a generator is ---------------------------------------------------

def test_param_maps_compose_by_substitution():
    """
    An ordinary parametric map is a morphism of ``Para``: it composes, and
    the parameter objects go side by side, left then right.  ``Dim`` is a
    strict monoid, so the layout is associative on the nose and a
    bracketing can never change which weights are where.
    """
    f = ParamMap("f", Dim(2), Dim(3), Dim(6))
    g = ParamMap("g", Dim(3), Dim(4), Dim(12))
    h = ParamMap("h", Dim(4), Dim(5), Dim(20))
    assert (f.name, f.dom, f.cod, f.params, f.laws) \
        == ("f", Dim(2), Dim(3), Dim(6), ())
    assert isinstance(f, Parametric) and ParamMap.id(Dim(2)).params == Dim()
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
    f = InteractionMap("f", Dim(2, 3), Dim(4, 5, 6))
    g = InteractionMap("g", Dim(7), Dim(8))
    with pytest.raises(AxiomError, match="do not compose"):
        f >> g
    with pytest.raises(TypeError):
        ParamMap("f", Dim(2), Dim(3)) @ g
    assert ParamMap("f", Dim(2), Dim(3)) \
        != InteractionMap("f", Dim(2), Dim(3))

    # the boundary is ``X* @ Y`` in the port order the module reads
    assert f.boundary == Dim(2, 3, 4, 5, 6) and f.width == 20
    assert f.dagger().boundary == Dim(4, 5, 6, 2, 3)
    assert f.dagger().width == 20 and f.dagger().dagger() == f
    # the tensor interleaves, so it is not the tensor of the boundaries
    assert (f @ g).boundary == Dim(2, 3, 7, 4, 5, 6, 8)
    assert (f @ g).boundary != f.boundary @ g.boundary
    assert (f @ g).width == f.width + g.width


def test_interaction_spec_reads_a_network():
    torch.manual_seed(0)
    module = torch.nn.Linear(5, 5)
    f = Network("f", Dim(2), Dim(3), module=module)
    spec = interaction_spec(f)
    assert spec == InteractionMap("f", Dim(2), Dim(3), Dim(30))
    assert spec.boundary == f.dom @ f.cod
    assert spec.width == module.in_features == module.out_features
    assert interaction_spec(f.dagger()) == spec.dagger()
    assert interaction_spec(Network("g", Dim(2), Dim(3))).params == Dim()
    assert interaction_spec(
        Network("g", Dim(2), Dim(3), module=object())).params == Dim()


def test_interaction_spec_changes_nothing():
    """
    Reading a spec registers nothing, wraps nothing and owns nothing: the
    module, its parameters, its training mode and the equality of the
    network are all exactly as they were.
    """
    module, found = compiled()
    before = {key: value.clone()
              for key, value in module.state_dict().items()}
    names = [name for name, _ in module.named_parameters()]
    one = Network("f", Dim(2), Dim(3), module=module)
    other = Network("f", Dim(2), Dim(3), module=module)
    assert one == other and hash(one) == hash(other)

    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        expected = found.advance(state, 3)

    specs = found.local
    assert [spec.name for spec in specs] == ["cell", "cell", "cell"]
    assert {spec.params for spec in specs} == {
        Dim(sum(p.numel() for p in module.parameters()))}

    assert list(module.state_dict()) == list(before)
    assert [name for name, _ in module.named_parameters()] == names
    assert module.training
    for key, value in module.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert one == other and hash(one) == hash(other)
    with torch.no_grad():
        assert torch.equal(found.advance(state, 3), expected)


# --- what a diagram compiles to --------------------------------------------

def test_the_interaction_agrees_with_the_map():
    module, found = compiled()
    cmap = found.cmap
    assert found.widths == cmap.port_widths
    assert found.routing == tuple(cmap.edges)
    assert found.state == Dim(*cmap.port_widths)
    assert found.total == cmap._routing["total"] == sum(cmap.port_widths)
    assert found.n_wires == cmap.n_ports // 2 == len(cmap.ports) // 2
    assert found.is_involution()
    assert [spec.name for spec in found.local] \
        == [box.name for box in cmap.boxes]
    for index, (spec, box) in enumerate(zip(found.local, cmap.boxes)):
        assert spec.boundary == box.dom @ box.cod
        assert spec.width == sum(
            cmap.port_widths[port] for port in cmap.box_ports(index))
    # sites share one module, so theta is *not* the product of their
    # parameter objects: the map has one module's worth of weights.
    assert sum(p.numel() for p in cmap.parameters()) \
        == sum(p.numel() for p in module.parameters())
    assert "3 boxes" in repr(found)


def test_a_map_keeps_the_shape_it_was_built_with():
    """
    :attr:`CMap.port_widths` is a ``cached_property``, which is sound
    exactly as long as a map's boxes are what its constructor fixed.  A
    map mutated afterwards would go on reading the table it built the
    first time it was asked, and that is the one way the cache can be
    wrong: silently, with an arithmetic answer of the right shape.

    So the assertion is not that the cache is fast, it is that the
    premise holds -- the boxes, the ports, the wiring and the widths
    recomputed from scratch are identical after a map has been advanced,
    asked for a residual, and advanced again.
    """
    _, found = compiled()
    cmap = found.cmap

    def widths():
        """ :attr:`CMap.port_widths` again, from the ports as they are. """
        return tuple(sum(getattr(port.obj, "inside", (port.obj, )))
                     for port in cmap.ports)

    def shape():
        return widths(), tuple(cmap.boxes), tuple(cmap.ports), \
            tuple(cmap.edges)

    before = shape()
    assert cmap.port_widths == before[0]
    state = torch.rand(2, sum(cmap.port_widths)).double()
    with torch.no_grad():
        for _ in range(3):
            state = found.advance(state, 1)
            found.residual(state)
    assert shape() == before
    assert cmap.port_widths == widths()


def test_only_a_closed_diagram_compiles():
    f = Network("f", Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    with pytest.raises(ValueError, match="closed"):
        Interaction(f.to_map())


def test_heads_are_read_off_the_wiring():
    """
    A port is a head unless it is wired to an earlier port of the same box,
    which is exactly the second copy of a traced leg.  No declaration is
    consulted: the wiring says which ports a module reads a value off.
    """
    _, found = compiled(((1, ), (0, )))
    assert found.ports["cell", STATE] == (1, 0, 4, 3)
    assert found.heads["cell", STATE] == (1, 4)
    assert found.ports["cell", PEER] == found.heads["cell", PEER] == (2, 5)
    assert found.sites(("cell", STATE)) == 2


def test_erasing_a_role_erases_its_wires():
    """
    ``Dim(0)`` is the monoidal unit, so a role sent to it leaves neither a
    port nor a wire -- which is how one diagram serves two models.
    """
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    kept = interpret(pair, {PEER: Dim(3), STATE: Dim(5)}, {"cell": None})
    erased = interpret(pair, {PEER: Dim(3), STATE: Dim(0)}, {"cell": None})
    assert kept.widths == (5, 5, 3, 5, 5, 3) and erased.widths == (3, 3)
    assert ("cell", STATE) not in erased.heads


# --- the formulae the docstrings state -------------------------------------

def test_round_is_routing_after_interaction():
    """
    ``T(s) = sigma(Phi(s))``: one round with no reinjection is the boxes'
    emissions carried along the wires, and :func:`route` is the readable
    spelling of that same permutation.
    """
    _, found = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        emitted = found.cmap(init=state, n_rounds=1, inject=False)
        routed = found.advance(state, 1)
    assert torch.equal(torch.cat(route(found.cmap, emitted), -1), routed)
    assert torch.equal(found.route(emitted), routed)


def test_reinjection_is_an_affine_shift():
    """
    ``T(s) = sigma(Phi(s)) + i``: the initial vector is added back to the
    *whole* state after routing, which is what ``inject=True`` means.
    """
    _, found = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        plain = found.advance(state, 1, inject=False)
        injected = found.advance(state, 1, inject=True)
    assert torch.equal(injected, plain + state)


def test_iteration_is_resumption():
    """
    ``T^(a+b) = T^b . T^a``, bitwise, which is what a segmented solver
    relies on -- and it holds for *one* transition, so a run resumed from
    its own carried state only resumes when ``inject`` is off.
    """
    _, found = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        whole = found.advance(state, 5)
        resumed = found.advance(found.advance(state, 2), 3)
        injected = found.advance(state, 5, inject=True)
        piecewise = found.advance(
            found.advance(state, 2, inject=True), 3, inject=True)
    assert torch.equal(whole, resumed)
    assert not torch.equal(injected, piecewise)


def test_a_fixed_point_is_a_fourth_thing():
    """
    Nothing about the category makes ``T`` contract.  The residual is a
    number to be measured: a zero map settles at once, a learned one is
    under no such obligation.
    """
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    still = torch.nn.Linear(3, 3).double()
    with torch.no_grad():
        still.weight.zero_(), still.bias.zero_()
    found = interpret(pair, {PEER: Dim(1), STATE: Dim(1)}, {"cell": still})
    state = found.zeros(2, dtype=torch.double)
    with torch.no_grad():
        settled, _ = Iterate(rounds=4, inject=False).run(found, state)
        assert float(found.residual(settled).max()) == 0.0

    torch.manual_seed(0)
    module, moving = compiled()
    torch.manual_seed(1)
    state = torch.randn(2, moving.total, dtype=torch.double)
    with torch.no_grad():
        assert float(moving.residual(state).max()) > 1e-3


def test_a_product_of_diagrams_is_a_product_of_states():
    """
    A batch of instances is the monoidal product of their maps, and the
    state of the product is the sum of the states: one summand per port,
    the members' ports in order.
    """
    _, pair = compiled(((1, ), (0, )))
    _, path = compiled(((1, ), (0, 2), (1, )))
    node = node_signature(1)
    torch.manual_seed(0)
    module = Site(node, {PEER: 3, STATE: 4}, {STATE: Mode.STATE},
                  hidden=6).double()
    both = interpret(
        from_relation(((1, ), (0, )), node)
        @ from_relation(((1, ), (0, 2), (1, )), node),
        {PEER: Dim(3), STATE: Dim(4)}, {"cell": module})
    assert both.total == pair.total + path.total
    assert both.widths == pair.widths + path.widths
    assert len(both.local) == 5


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


# --- laws: the actions agree with Sym --------------------------------------

def test_actions_agree_with_sym():
    unit = Signature((Orbit(MESSAGE, 3, Sym.PERM),
                      Orbit(STATE, traced=True)))
    found = laws.symmetry(unit)
    assert len(found) == 1
    law, = found
    assert law.roles == (MESSAGE, )
    assert law.action == laws.Action(Sym.PERM, 3)
    assert law.action.generators == ((1, 0, 2), (1, 2, 0))
    assert law.action.order == 6 and not law.action.is_trivial
    assert law.strictness == laws.Strictness.LAX
    assert str(law) == "message is lax perm-equivariant over 3 legs"
    assert sum(len(one.action.generators) for one in found) \
        == len(unit.generators())


@pytest.mark.parametrize("sym,degree,order,generators", [
    (Sym.NONE, 4, 1, ()),
    (Sym.CYCLIC, 4, 4, ((1, 2, 3, 0), )),
    (Sym.PERM, 4, 24, ((1, 0, 2, 3), (1, 2, 3, 0))),
    (Sym.PERM, 1, 1, ()),
])
def test_action_of_every_sym(sym, degree, order, generators):
    action = laws.action(Orbit(MESSAGE, degree, sym))
    assert action.generators == generators
    assert action.order == order
    assert action.is_trivial == (not generators)
    declared = Signature((Orbit(MESSAGE, degree, sym), ))
    assert bool(laws.symmetry(declared)) == bool(generators)


def test_laws_are_the_equations_measured():
    """
    A law is not a claim: it names the group :func:`check_equivariant` runs
    the module against.  A pooled relation keeps it laxly, a module with
    one weight per port does not keep it at all -- and keeping the symmetry
    is *not* keeping Frobenius.
    """
    unit = Signature((Orbit(MESSAGE, 4, Sym.PERM), ))
    law, = laws.symmetry(unit, strictness=laws.Strictness.LAX)
    torch.manual_seed(0)
    relation = Relation(unit, {MESSAGE: 2}, hidden=4).double()
    assert check_equivariant(relation, unit, {MESSAGE: 2})[MESSAGE] < 1e-12

    class Skew(torch.nn.Module):
        """ Weighs each port differently, so it cannot commute. """
        def forward(self, x):
            return x * torch.arange(1., 1 + x.shape[-1], dtype=x.dtype)

    with pytest.raises(AxiomError, match=str(law.action.group)):
        check_equivariant(Skew(), unit, {MESSAGE: 2})
    assert fusion_residual(relation, unit, {MESSAGE: 2}) > 1e-3


def test_a_site_carries_its_laws():
    """
    A spec can carry the laws of the site it fills, which is the only place
    the two halves of the semantics meet.
    """
    node = node_signature(3)
    torch.manual_seed(0)
    module = Site(node, {PEER: 3, STATE: 4}, {STATE: Mode.STATE},
                  hidden=6).double()
    network = Network("cell", Dim(), Dim(3, 3, 3, 4, 4), module=module)
    spec = interaction_spec(network, laws.symmetry(node))
    assert [str(law) for law in spec.laws] \
        == ["peer is lax perm-equivariant over 3 legs"]
    assert spec.width == node.width({PEER: 3, STATE: 4}) == 17
    assert check_equivariant(module, node, {PEER: 3, STATE: 4})[PEER] < 1e-12
    # the tensor forgets the laws, because a product makes no such promise
    assert (spec @ InteractionMap("g", Dim(2), Dim(2))).laws == ()
