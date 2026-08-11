# -*- coding: utf-8 -*-

"""
The formal layer: what the executable code *means*, checked against the
executable code.

Nothing here is a second numerical backend, so nothing here asserts a
tensor value on its own.  What it asserts is agreement: that
:func:`~discopy.neural.parametric.interaction_spec` reproduces the boundary
of a :class:`~discopy.neural.Network` without touching it, that
:func:`~discopy.neural.dynamics.from_map` reproduces the state width and the
routing of a :class:`~discopy.neural.CMap`, that the formulae the docstrings
state -- ``T(s) = sigma(Phi(s)) + i`` and ``T^(a+b) = T^b . T^a`` -- are the
ones the forward pass computes, and that a law of
:mod:`~discopy.neural.laws` is the symmetry
:func:`~discopy.neural.signature.check_equivariant` measures.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from discopy.frobenius import Ty
from discopy.neural import Dim, Network, Orbit, Signature, Sym
from discopy.neural import laws
from discopy.neural.cells import Mode, Relation, Site
from discopy.neural.dynamics import Iteration, Transition, from_map
from discopy.neural.functor import Interpretation, interpret, route
from discopy.neural.parametric import (
    InteractionMap, ParamMap, Parametric, interaction_spec)
from discopy.neural.signature import check_equivariant
from discopy.neural.skeleton import from_relation
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


def wiring_of(relation=((1, ), (0, 2), (1, ))):
    """ A path graph of sites sharing one module, in float64. """
    node = node_signature(1)
    torch.manual_seed(0)
    module = Site(node, {PEER: 3, STATE: 4}, {STATE: Mode.STATE},
                  hidden=6).double()
    return module, interpret(Interpretation(
        {PEER: Dim(3), STATE: Dim(4)}, {"cell": module}),
        from_relation(relation, node))


# --- the formal layer is torch-free ----------------------------------------

def test_formal_layer_does_not_import_torch():
    """
    The three formal modules are specifications, so they load on a machine
    with no torch at all -- as ``discopy.neural`` itself already did.
    """
    script = ("import sys, discopy.neural;"
              " from discopy.neural import dynamics, laws, parametric;"
              " print('torch' in sys.modules)")
    found = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert found.stdout.strip() == "False"


# --- parametric: metadata and the ordering of parameters -------------------

def test_param_metadata():
    f = ParamMap("f", Dim(2), Dim(3), Dim(6))
    assert (f.name, f.dom, f.cod, f.params) == ("f", Dim(2), Dim(3), Dim(6))
    assert f.laws == ()
    assert ParamMap.id(Dim(2)).params == Dim()


def test_param_ordering_is_deterministic():
    """
    Composition and tensor lay the parameter objects out left then right,
    and the layout is associative on the nose because ``Dim`` is a strict
    monoid -- so a bracketing can never change which weights are where.
    """
    f = ParamMap("f", Dim(2), Dim(3), Dim(6))
    g = ParamMap("g", Dim(3), Dim(4), Dim(12))
    h = ParamMap("h", Dim(4), Dim(5), Dim(20))
    assert (f >> g).params == Dim(6, 12) != (g.params @ f.params)
    assert ((f >> g) >> h).params == (f >> (g >> h)).params == Dim(6, 12, 20)
    assert ((f @ g) @ h).params == (f @ (g @ h)).params == Dim(6, 12, 20)
    assert ((f @ g) @ h).dom == (f @ (g @ h)).dom == Dim(2, 3, 4)
    for one, other in (((f >> g) >> h, f >> (g >> h)),
                       ((f @ g) @ h, f @ (g @ h))):
        assert (one.dom, one.cod, one.params) \
            == (other.dom, other.cod, other.params)


def test_param_composition_is_typed():
    f = ParamMap("f", Dim(2), Dim(3))
    with pytest.raises(AxiomError, match="does not compose"):
        f >> f
    # the two readings never mix: an interaction is not a feed-forward map
    with pytest.raises(TypeError):
        f >> InteractionMap("g", Dim(3), Dim(4))
    with pytest.raises(TypeError):
        InteractionMap("g", Dim(3), Dim(4)) @ f
    assert ParamMap("f", Dim(2), Dim(3)) \
        != InteractionMap("f", Dim(2), Dim(3))
    assert isinstance(f, Parametric)


def test_interaction_boundary():
    """
    The boundary of an interaction is ``X* @ Y`` in the port order the
    module reads, and its width is the module's width.
    """
    f = InteractionMap("f", Dim(2, 3), Dim(4, 5, 6))
    assert f.boundary == Dim(2, 3, 4, 5, 6) and f.width == 20
    # the dual is the same object up to the symmetry, hence the same width
    assert f.dagger().boundary == Dim(4, 5, 6, 2, 3) and f.dagger().width == 20
    assert f.dagger().dagger() == f
    # the tensor interleaves, so it is not the tensor of the boundaries
    g = InteractionMap("g", Dim(7), Dim(8))
    assert (f @ g).boundary == Dim(2, 3, 7, 4, 5, 6, 8)
    assert (f @ g).boundary != f.boundary @ g.boundary
    assert (f @ g).width == f.width + g.width


# --- interaction_spec reads a Network without touching it ------------------

def test_interaction_spec_reads_a_network():
    torch.manual_seed(0)
    module = torch.nn.Linear(5, 5)
    f = Network("f", Dim(2), Dim(3), module=module)
    spec = interaction_spec(f)
    assert spec == InteractionMap("f", Dim(2), Dim(3), Dim(30))
    assert spec.boundary == f.dom @ f.cod
    assert spec.width == module.in_features == module.out_features
    assert interaction_spec(f.dagger()) == spec.dagger()
    # a network with no module, or with data that is not a torch module
    assert interaction_spec(Network("g", Dim(2), Dim(3))).params == Dim()
    assert interaction_spec(
        Network("g", Dim(2), Dim(3), module=object())).params == Dim()


def test_interaction_spec_changes_nothing():
    """
    Reading a spec registers nothing, wraps nothing and owns nothing: the
    module, its parameters, its training mode and the equality of the
    network are all exactly as they were.
    """
    module, wiring = wiring_of()
    before = {key: value.clone() for key, value in module.state_dict().items()}
    names = [name for name, _ in module.named_parameters()]
    one = Network("f", Dim(2), Dim(3), module=module)
    other = Network("f", Dim(2), Dim(3), module=module)
    assert one == other and hash(one) == hash(other)

    torch.manual_seed(1)
    state = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        expected = wiring.cmap(init=state, n_rounds=3, return_flat=True)

    specs = [interaction_spec(box) for box in wiring.cmap.boxes]
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
        assert torch.equal(
            wiring.cmap(init=state, n_rounds=3, return_flat=True), expected)


# --- dynamics: the transition agrees with the map --------------------------

def test_transition_agrees_with_the_map():
    module, wiring = wiring_of()
    cmap = wiring.cmap
    round_ = from_map(cmap)
    assert round_.widths == cmap.port_widths
    assert round_.routing == tuple(cmap.edges)
    assert round_.state == Dim(*cmap.port_widths)
    assert round_.width == cmap._routing["total"] == wiring.router.total
    assert round_.n_ports == cmap.n_ports == len(cmap.ports)
    assert round_.is_closed and round_.is_involution()
    assert round_.inject and not from_map(cmap, inject=False).inject
    assert [spec.name for spec in round_.local] \
        == [box.name for box in cmap.boxes]
    for index, (spec, box) in enumerate(zip(round_.local, cmap.boxes)):
        assert spec.boundary == box.dom @ box.cod
        assert spec.width == sum(
            cmap.port_widths[port] for port in cmap.box_ports(index))
    # sites share one module, so theta is *not* the product of their
    # parameter objects: the map has one module's worth of weights.
    assert sum(p.numel() for p in cmap.parameters()) \
        == sum(p.numel() for p in module.parameters())


def test_transition_of_an_open_map():
    f = Network("f", Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    round_ = from_map(f.to_map())
    assert not round_.is_closed
    assert round_.inputs == (0, ) and round_.outputs == (3, )
    assert round_.widths == (2, 2, 3, 3) and round_.width == 10
    assert [spec.width for spec in round_.local] == [5]


def test_transition_involution_is_checked():
    assert Transition((), (1, 0), (2, 2)).is_involution()
    assert not Transition((), (0, 1), (2, 2)).is_involution()
    assert not Transition((), (1, 0, 3, 2), (1, ) * 4,
                          inputs=(0, )).is_closed


# --- dynamics: the formulae the docstrings state ---------------------------

def test_round_is_routing_after_interaction():
    """
    ``T(s) = sigma(Phi(s))``: one round with no reinjection is the boxes'
    emissions carried along the wires, and :func:`route` is the readable
    spelling of that same permutation.
    """
    _, wiring = wiring_of()
    torch.manual_seed(1)
    state = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        emitted = wiring.cmap(init=state, n_rounds=1, inject=False)
        routed = wiring.cmap(
            init=state, n_rounds=1, inject=False, return_flat=True)
    assert torch.equal(torch.cat(route(wiring.cmap, emitted), -1), routed)
    assert torch.equal(wiring.router(emitted), routed)


def test_reinjection_is_an_affine_shift():
    """
    ``T(s) = sigma(Phi(s)) + i``: the initial vector is added back to the
    *whole* state after routing, which is what ``inject=True`` means.
    """
    _, wiring = wiring_of()
    torch.manual_seed(1)
    state = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        plain = wiring.cmap(
            init=state, n_rounds=1, inject=False, return_flat=True)
        injected = wiring.cmap(
            init=state, n_rounds=1, inject=True, return_flat=True)
    assert torch.equal(injected, plain + state)


def test_iteration_is_resumption():
    """
    ``T^(a+b) = T^b . T^a``: the law :class:`Iteration` records is the one
    the forward pass computes, bitwise, which is what a segmented loop
    relies on.
    """
    _, wiring = wiring_of()
    round_ = from_map(wiring.cmap, inject=False)
    assert round_.iterate(2) >> round_.iterate(3) == round_.iterate(5)
    assert round_.iterate(0).rounds == 0
    torch.manual_seed(1)
    state = torch.randn(2, wiring.router.total, dtype=torch.double)
    kwargs = dict(inject=False, return_flat=True)
    with torch.no_grad():
        whole = wiring.cmap(init=state, n_rounds=5, **kwargs)
        first = wiring.cmap(init=state, n_rounds=2, **kwargs)
        resumed = wiring.cmap(init=first, n_rounds=3, **kwargs)
    assert torch.equal(whole, resumed)


def test_reinjection_belongs_to_the_transition():
    """
    With ``inject=True`` the initial vector ``i`` is part of ``T``, so a
    run resumed from the carried state resumes a *different* transition.
    The engine never does that: a schedule with ``inject=True`` runs one
    ``advance`` per forward pass, and the segmented schedules that resume
    pass ``inject=False``.
    """
    _, wiring = wiring_of()
    torch.manual_seed(1)
    state = torch.randn(2, wiring.router.total, dtype=torch.double)
    kwargs = dict(inject=True, return_flat=True)
    with torch.no_grad():
        whole = wiring.cmap(init=state, n_rounds=5, **kwargs)
        first = wiring.cmap(init=state, n_rounds=2, **kwargs)
        resumed = wiring.cmap(init=first, n_rounds=3, **kwargs)
    assert not torch.equal(whole, resumed)


def test_iteration_composition_is_typed():
    round_ = Transition((), (1, 0), (2, 2))
    with pytest.raises(ValueError, match="different transitions"):
        round_.iterate(1) >> Transition((), (1, 0), (4, 4)).iterate(1)
    with pytest.raises(TypeError):
        round_.iterate(1) >> 2
    with pytest.raises(ValueError):
        Iteration(round_, -1)


# --- laws: the actions agree with Sym --------------------------------------

def test_actions_agree_with_sym():
    unit = Signature((Orbit(MESSAGE, 3, Sym.PERM), Orbit(STATE, traced=True)))
    found = laws.symmetry(unit)
    assert len(found) == 1
    law, = found
    assert law.roles == (MESSAGE, )
    assert law.action == laws.Action(Sym.PERM, 3)
    assert law.action.generators == ((1, 0, 2), (1, 2, 0))
    assert law.action.order == 6 and not law.action.is_trivial
    assert law.strictness == laws.Strictness.LAX
    assert str(law) == "message is lax perm-equivariant over 3 legs"
    # one law per orbit, and its generators are the signature's, counted
    # at the level of legs rather than of ports
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
    A law is not a claim: it names the group
    :func:`check_equivariant` runs the module against.  A pooled relation
    keeps it laxly, a module with one weight per port does not keep it at
    all.
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

    # and the honest part: keeping the symmetry is not keeping Frobenius
    from discopy.neural.cells import fusion_residual
    assert fusion_residual(relation, unit, {MESSAGE: 2}) > 1e-3


def test_a_site_carries_its_laws():
    """
    A spec can carry the laws of the site it fills, which is the only
    place the two halves of the formal layer meet.
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
    # composition forgets the laws, because a composite makes no such
    # promise about the ports that were glued
    assert (spec >> InteractionMap("g", spec.cod, Dim(2))).laws == ()


# --- the map is a monoidal product, and so is its transition ---------------

def test_transition_of_a_product():
    """
    A batch of instances is the monoidal product of their maps, and the
    state of the product is the sum of the states: one summand per port,
    the members' ports in order.
    """
    _, pair = wiring_of(((1, ), (0, )))
    _, path = wiring_of(((1, ), (0, 2), (1, )))
    node = node_signature(1)
    torch.manual_seed(0)
    module = Site(node, {PEER: 3, STATE: 4}, {STATE: Mode.STATE},
                  hidden=6).double()
    both = interpret(
        Interpretation({PEER: Dim(3), STATE: Dim(4)}, {"cell": module}),
        from_relation(((1, ), (0, )), node)
        @ from_relation(((1, ), (0, 2), (1, )), node))
    assert from_map(both.cmap).width \
        == from_map(pair.cmap).width + from_map(path.cmap).width
    assert from_map(both.cmap).widths \
        == from_map(pair.cmap).widths + from_map(path.cmap).widths
    assert len(from_map(both.cmap).local) == 5


def test_a_snake_is_pure_rerouting():
    """
    The transition of a diagram with no boxes has no local interaction at
    all: swaps, cups and caps are wiring, which a functor preserves
    strictly and for free.
    """
    from discopy.neural import Id
    snake = Id(Dim(2)).transpose().to_map()
    round_ = from_map(snake)
    assert round_.local == () and snake.boxes == ()
    assert round_.widths == (2, 2) and round_.width == 4
    assert round_.inputs == (0, ) and round_.outputs == (1, )
