from pytest import raises

from discopy.hypergraph import *
from discopy.frobenius import Ty, Box, Cap, Hypergraph as H

def test_pushout():
    with raises(ValueError):
        pushout(1, 1, [0], [0, 1])


def test_Hypergraph_init():
    x, y = map(Ty, "xy")
    with raises(ValueError):
        H(x, x, (), ())
    with raises(AxiomError):
        H(x, y, (), ((0,), (), (0,)))


def test_Hypergraph_str():
    x, y = map(Ty, "xy")
    assert str(H.swap(x, y)) == "Swap(x, y)"
    assert str(H.spiders(1, 0, x @ y))\
        == "Spider(1, 0, x) @ Spider(1, 0, y)"


def test_Hypergraph_repr():
    x, y = map(Ty, "xy")
    assert repr(H.spiders(1, 0, x @ y))\
        == "hypergraph.Hypergraph[Diagram]("\
           "dom=frobenius.Ty(frobenius.Ob('x'), frobenius.Ob('y')), "\
           "cod=frobenius.Ty(), boxes=(), wires=((0, 1), (), ()))"


def test_Hypergraph_hash():
    x, y = map(Ty, "xy")
    assert hash(H.id(x @ y)) == hash(H.id(x) @ H.id(y))


def test_Hypergraph_then():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.id(x) >> H.id(y)


def test_Hypergraph_tensor():
    Id = H.id
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Hypergraph_getitem():
    with raises(NotImplementedError):
        H.spiders(1, 2, Ty('x'))[0]


def test_Hypergraph_bijection():
    with raises(ValueError):
        H.spiders(1, 2, Ty('x')).bijection


def test_Hypergraph_make_causal_does_not_assume_topological_order():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', x, y), Box('g', y, z)
    h = H(
        x, z, (g, f),
        ((0,), (((1,), (2,)), ((0,), (1,))), (2,)))

    assert h.is_left_monogamous
    assert h.is_acyclic
    assert not h.is_topologically_ordered
    assert not h.is_causal
    assert h.make_causal().is_causal
    assert h.topological_order().is_causal


def test_Hypergraph_simplify_bubble_size():
    x = Ty('x')
    f = Box('f', Ty(), x)
    g, h = Box('g', x, x), Box('h', x, Ty())
    diagram = f >> (g >> h).bubble()
    hypergraph = diagram.to_hypergraph()
    interchanged = hypergraph.interchange(0, 1)

    assert len(diagram) == 2 and diagram.size == 4
    assert interchanged.to_diagram().size > diagram.size
    assert interchanged.simplify() == hypergraph


def test_Hypergraph_rotate():
    assert H.id() == \
           H.id().rotate(left=False).rotate(left=True)


def test_Box():
    box = Box('box', Ty('x'), Ty('y')).to_hypergraph()
    assert box == box and box == box @ H.id() and box != 1


def test_AxiomError():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.cups(x @ y, x @ y)
    with raises(AxiomError):
        H.caps(x @ y, x @ y)


def test_non_adjoint_wire():
    from discopy import compact
    x = compact.Ty('x')
    K = compact.Hypergraph
    # A cap with non-adjoint legs (x, x) is rejected at construction.
    with raises(AxiomError):
        K(compact.Ty(), x @ x, (), ((), (), (0, 0)))
    # Adjoint cups and caps, and self-dual frobenius caps, are fine.
    assert K.cups(x, x.r) and K.caps(x.r, x)
    assert Cap(Ty('x'), Ty('x')).to_hypergraph()


def test_cups():
    x = Ty('x')
    assert H.cups(x, x).make_monogamous().dagger()\
        == H.caps(x, x).make_monogamous()
    assert H.caps(x, x).make_monogamous().dagger()\
        == H.cups(x, x).make_monogamous()
    assert H.caps(x, x).to_diagram() == Cap(x, x)


def test_Hypergraph_is_boundary_connected():
    x, y, z = map(Ty, "xyz")
    f = Box('f', x, x).to_hypergraph()
    assert f.is_boundary_connected

    g = Box('g', x @ z, y @ z).to_hypergraph()
    assert g.trace().is_boundary_connected

    assert not f.trace().is_boundary_connected

    scalar = Box('s', Ty(), Ty()).to_hypergraph()
    assert not (scalar @ scalar).is_boundary_connected
    assert not H.spiders(0, 0, x).is_boundary_connected

    assert H.id(Ty()).is_boundary_connected
    assert H.id(x).is_boundary_connected


def test_Hypergraph_eq_fast_path_trace():
    """ Traces are cyclic but boundary-connected, so they should use the
    fast canonical-form path rather than falling back to VF2. """
    x, y, z = map(Ty, "xyz")
    g = Box('g', x @ z, y @ z).to_hypergraph()
    trace = g.trace()

    assert trace.is_acyclic is False
    assert trace.is_fast_eligible

    shuffled = trace.interchange(0, min(1, len(trace.boxes) - 1))
    assert trace == shuffled
    assert hash(trace) == hash(shuffled)


def test_Hypergraph_eq_fallback_scalars_and_empty_boundary():
    """ Hypergraphs that are not boundary-connected (scalars, closed
    diagrams) must still compare correctly via the VF2 fallback. """
    scalar_a = Box('s', Ty(), Ty()).to_hypergraph()
    scalar_b = Box('s', Ty(), Ty()).to_hypergraph()
    assert not (scalar_a @ scalar_b).is_fast_eligible
    assert scalar_a @ scalar_b == scalar_b @ scalar_a

    x = Ty('x')
    f = Box('f', x, x).to_hypergraph()
    assert f.trace() == f.trace()


def test_simplify():
    from discopy.markov import Diagram, Box, Ty, Copy, Swap, Trace
    C, T, P = map(Ty, "CTP")
    linear, param_linear, add, placeholder = (
        Box('linear', T @ P, T),
        Box('param_linear', C, P),
        Box('add', T @ T, T),
        Box('placeholder', C, T),
    )
    residual_block = Trace(Trace(Copy(C) @ T @ P >> C @ C @ linear >> param_linear @ C @ T >> P @ placeholder @ T >> P @ Copy(T) @ T >> P @ T @ Swap(T, T) >> P @ add @ T >> Swap(P, T) @ T >> T @ Swap(P, T)))
    ref = Copy(C) >> param_linear @ C >> P @ placeholder >> P @ Copy(T) >> Swap(P, T) @ T >> linear @ T >> Swap(T, T) >> add
    simpl = residual_block.to_hypergraph().simplify().to_diagram()

    with Diagram.hypergraph_equality:
        assert residual_block == ref == simpl

    # to_diagram foliates, so simpl is the (tighter) foliation of ref rather
    # than its point-free, one-box-per-layer presentation.
    assert simpl == ref.foliation()


def test_parameterisation():
    from discopy import frobenius
    assert H == Hypergraph[frobenius.Diagram]
    assert H.category == frobenius.Diagram
    assert H.functor == frobenius.Functor == H.category.functor_factory
    assert H.ob == frobenius.Ty


def test_subclass_to_hypergraph():
    from discopy import frobenius
    from discopy.cat import factory

    @factory
    class Circuit(frobenius.Diagram):
        """ A frobenius diagram with hypergraphs of its own category. """

    class Gate(frobenius.Box, Circuit):
        """ A gate is a box in a circuit. """

    x = Ty('x')
    f, g = Gate('f', x, x), Gate('g', x, x)
    assert (f >> g).to_hypergraph().category == Circuit
    assert isinstance((f >> g).to_hypergraph().to_diagram(), Circuit)
    with Circuit.hypergraph_equality:
        assert f >> g == (f >> g)
        assert hash(f >> g) == hash(f >> g)
