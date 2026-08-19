import shutil

import pytest
from pytest import raises

from discopy import closed, biclosed, compact, symmetric
from discopy.python.finset import Permutation
from discopy.utils import AxiomError


def test_port_side_and_depth():
    from discopy.compact import Ty, Box, CMap as M
    x = Ty("x")
    ports = M.id(x).ports
    assert ports[0].side == "up"
    assert ports[1].side == "down"
    assert [port.depth for port in ports] == [-float("inf"), float("inf")]
    adjoint_ports = M.id(x.r).ports
    assert adjoint_ports[0].side == "up"
    assert adjoint_ports[1].side == "down"
    f = Box("f", x, x)
    box_ports = (M.from_box(f) >> M.from_box(f)).ports
    assert [port.depth for port in box_ports] == [
        -float("inf"), 0.5, -0.5, 1.5, 0.5, float("inf")]


def test_default_compact_setting():
    from discopy.compact import Ty, Box, CMap as M
    x, y = map(Ty, "xy")
    f = Box("f", x, y)
    cm = M.from_box(f)
    assert isinstance(f, M.category)
    assert cm.to_hypergraph().category == M.category


def test_M_init():
    from discopy.compact import Ty, CMap as M
    x, y = map(Ty, "xy")
    with raises(ValueError):
        M(x, x, (), ())
    with raises(ValueError):
        M(x, x, (), (0, 1))
    with raises(ValueError):
        M(x, x, (), (0,))
    with raises(AxiomError):
        M(x @ y, x @ y, (), (1, 0, 3, 2))
    with raises(AxiomError):
        M(x, y, (), (1, 0))


def test_repr_eq_and_hash():
    from discopy.compact import Ty, Box, CMap as M

    x, y = map(Ty, "xy")
    cm = M.from_box(Box("f", x, y))
    with_metadata = M(cm.dom, cm.cod, cm.boxes, cm.edges, loops=(x, ))
    namespace = {}
    exec("from discopy import *", namespace)
    back = eval(repr(with_metadata), namespace)
    assert back == with_metadata
    assert back.loops == with_metadata.loops
    assert cm == M.from_box(Box("f", x, y))
    assert cm != object()
    assert hash(cm) == hash(M.from_box(Box("f", x, y)))

    g = M.from_box(Box("g", y, x))
    interchanged = (cm @ g).interchange(0, 1)
    assert interchanged.boxes != (cm @ g).boxes
    assert (cm @ g).to_hypergraph() == interchanged.to_hypergraph()


def test_id_and_tensor():
    from discopy.compact import Ty, CMap as M, Hypergraph as H
    x, y = map(Ty, "xy")
    assert M.id(x).edges == (1, 0)
    assert M.id(x).orientation == (1, 0)
    assert M.id(x).faces == (0, 1)
    assert M.id().tensor() == M.id()
    assert M.id(x).tensor(M.id(y)) == M.id(x) @ M.id(y)
    assert (M.id(x) @ M.id(y)).to_hypergraph() == H.id(x @ y)


def test_from_box_and_to_hypergraph():
    from discopy.compact import Ty, Box, CMap as M
    x, y, z = map(Ty, "xyz")
    f = Box("f", x, y)
    cm = M.from_box(f)
    assert cm.edges == (1, 0, 3, 2)
    assert cm.orientation == (3, 2, 1, 0)
    assert cm.faces == (2, 3, 0, 1)
    assert cm.to_hypergraph() == f.to_hypergraph()

    multi_input = M.from_box(Box("g", x @ y, z))
    assert multi_input.orientation == Permutation.from_cycles(
        [(1, 0, 5), (2, 3, 4)], 6)


def test_eliminate_swaps():
    from discopy.compact import Ty, Id, Box

    x, y, w, z = map(Ty, "xyzw")

    diagram = Id(x @ y).swap(x, y).swap(y, x)
    assert diagram == diagram.to_map().to_diagram().normal_form()

    diagram = Id(x @ y @ w @ z)\
        .swap(x @ y, w @ z).swap(w @ z, x @ y).normal_form()
    assert diagram == diagram.to_map().to_diagram().normal_form()

    f, g = Box("f", x, z), Box("g", y, w)

    diagram = Id(x @ y).swap(x, y) >> g @ x >> Id(w @ x).swap(w, x) >> f @ w
    assert diagram.to_map().to_diagram() == x @ g >> f @ w
    assert diagram.to_map() == diagram.to_map().to_diagram().to_map()
    assert diagram.to_map() == diagram.to_hypergraph().to_diagram().to_map()


def test_states_decode_where_they_were():
    from discopy.symmetric import Ty, Box

    x, y = map(Ty, "xy")
    state = Box("s", Ty(), y)
    diagram = x @ state
    assert diagram.to_map().to_diagram() == diagram


def test_diagram_to_map():
    from discopy.monoidal import Ty, Box

    x, y, z = map(Ty, "xyz")
    f, g = Box("f", x, y), Box("g", y, z)
    assert (f >> g).to_map() == f.to_map() >> g.to_map()
    assert (f @ g).to_map() == f.to_map() @ g.to_map()


def test_symmetric_diagram_to_map_encodes_swap_as_wiring():
    from discopy import monoidal, symmetric

    x, y = map(symmetric.Ty, "xy")
    cm = symmetric.Id(x @ y).permute(1, 0).to_map()
    assert cm.dom == x @ y
    assert cm.cod == y @ x
    assert cm.boxes == ()
    assert cm.edges == (3, 2, 1, 0)

    x = symmetric.Ty("x")
    with raises(AxiomError):
        monoidal.CMap(x @ x, x @ x, (), (3, 2, 1, 0)).to_diagram()
    assert symmetric.CMap(x @ x, x @ x, (), (3, 2, 1, 0))\
        == symmetric.CMap.swap(x, x)

    x, y, z = map(monoidal.Ty, "xyz")
    f = monoidal.Box("f", x @ y, z)
    with raises(AxiomError):
        monoidal.CMap(y @ x, z, (f, ), (3, 2, 1, 0, 5, 4)).to_diagram()


def test_diagram_to_map_structure_and_errors():
    from discopy import (
        braided,
        closed,
        compact,
        frobenius,
        markov,
        monoidal,
        pivotal,
        symmetric,
        traced,
    )
    from discopy.cmap import Port, PortKind

    mx, my = map(monoidal.Ty, "xy")
    f = monoidal.Box("f", mx, my)
    assert f.to_map() == monoidal.CMap.from_box(f)

    bx, by = map(braided.Ty, "xy")
    braid = braided.Braid(bx, by)
    assert monoidal.CMap.from_diagram(braid).boxes == (braid, )

    sx, sy = map(symmetric.Ty, "xy")
    assert symmetric.Swap(sx, sy).to_map() == symmetric.CMap.swap(sx, sy)

    cx = compact.Ty("x")
    cup = compact.Cup(cx, cx.r)
    cap = compact.Cap(cx.r, cx)
    assert symmetric.CMap.from_diagram(cup).boxes == (cup, )
    assert cup.to_map() == compact.CMap.cups(cx, cx.r)
    assert cap.to_map() == compact.CMap.caps(cx.r, cx)

    tx = traced.Ty("x")
    traced_box = traced.Box("f", tx, tx)
    assert traced.Trace(traced_box).to_map() == traced_box.to_map().trace()

    px, py = map(pivotal.Ty, "xy")
    pbox = pivotal.Box("f", px, py)
    assert pbox.transpose(left=True).transpose(left=False).to_map().boxes\
        == (pbox, )

    cx, cy = map(closed.Ty, "xy")
    ev = closed.Eval(cy << cx)
    assert ev.to_map() == closed.CMap.ev(cy, cx, left=False)
    assert ev.to_map().boxes == (ev, )
    assert closed.Box("f", cx, cx).to_map().trace()

    mx = markov.Ty("x")
    copy = markov.Copy(mx, 2)
    assert copy.to_map() == markov.CMap.copy(mx, 2)

    fx = frobenius.Ty("x")
    spider = frobenius.Spider(1, 2, fx)
    assert markov.CMap.from_diagram(spider).boxes == (spider, )
    assert spider.to_map() == frobenius.CMap.spiders(1, 2, fx)

    x, y = map(compact.Ty, "xy")
    assert compact.CMap.swap(x, y).to_hypergraph()\
        == compact.CMap.category.swap(x, y).to_hypergraph()
    assert compact.CMap.cups(x, x.r).dom == x @ x.r
    assert compact.CMap.caps(x.r, x).cod == x.r @ x
    with raises(AxiomError):
        compact.CMap.cups(x, y)
    with raises(AxiomError):
        compact.CMap.caps(x, y)
    with raises(AxiomError):
        compact.CMap(x, x.r, (), (1, 0))
    with raises(AxiomError):
        compact.CMap.validate_wire(
            Port(PortKind.INPUT, 0, x, 0, "up"),
            Port(PortKind.COD, 0, x, 0, "down"))
    f = compact.CMap.from_box(compact.Box("f", x, y))
    assert f.trace(0) is f
    with raises(ValueError):
        f.trace(-1)
    with raises(ValueError):
        f.trace(2)

    x = monoidal.Ty("x")
    with raises(TypeError, match="Pregroup"):
        monoidal.CMap.cups(x, x)
    with raises(TypeError, match="Pregroup"):
        monoidal.CMap.caps(x, x)
    with raises(TypeError):
        compact.CMap.cups(x, x)
    with raises(AxiomError):
        monoidal.CMap(x @ x, monoidal.Ty(), (), (1, 0))
    with raises(AxiomError):
        monoidal.CMap(monoidal.Ty(), x @ x, (), (1, 0))
    assert monoidal.CMap.id(x).edges == (1, 0)
    f = monoidal.Box("f", x, x)
    g = monoidal.Box("g", x, x)
    with raises(AxiomError, match="has no traces"):
        monoidal.CMap(
            monoidal.Ty(), monoidal.Ty(), (f, g), (3, 2, 1, 0)).to_diagram()
    s = monoidal.Box("s", monoidal.Ty(), monoidal.Ty())
    t = monoidal.Box("t", monoidal.Ty(), monoidal.Ty())
    assert monoidal.CMap(
        monoidal.Ty(), monoidal.Ty(), (s, t), ()).to_diagram() == s >> t
    for module in [closed, traced, symmetric]:
        x = module.Ty("x")
        f = module.Box("f", x, x)
        g = module.Box("g", x, x)
        cycle = module.CMap(
            module.Ty(), module.Ty(), (f, g), (3, 2, 1, 0))
        assert not cycle.is_acyclic

    x = traced.Ty("x")
    with raises(TypeError, match="Pregroup"):
        traced.CMap.cups(x, x)
    with raises(TypeError, match="Pregroup"):
        traced.CMap.caps(x, x)
    x, y = map(closed.Ty, "xy")
    assert closed.CMap.ev(y, x).boxes == (
        closed.CMap.category.ev(y, x), )

    x = markov.Ty("x")
    assert markov.CMap.copy(x, 2).boxes == (
        markov.CMap.category.copy(x, 2), )
    assert markov.CMap.merge(x, 2).boxes == (
        markov.CMap.category.merge(x, 2), )
    assert markov.CMap.discard(x).boxes == (markov.CMap.category.copy(x, 0), )

    x = frobenius.Ty("x")
    assert frobenius.CMap.spiders(1, 2, x).boxes == (
        frobenius.Diagram.spiders(1, 2, x), )


def test_rigid_handedness():
    from discopy import cmap, rigid

    M = cmap.CMap[rigid.Diagram]
    x, y = map(rigid.Ty, "xy")

    assert M.cups(x, x.r).to_diagram() == rigid.Cup(x, x.r)
    assert M.caps(x.r, x).to_diagram() == rigid.Cap(x.r, x)
    with raises(AxiomError, match="pivotal"):
        M.cups(x.r, x)
    with raises(AxiomError, match="pivotal"):
        M.caps(x, x.r)
    with raises(AxiomError, match="has no swaps"):
        M.swap(x, y).to_diagram()

    cx = compact.Ty("x")
    assert compact.CMap.cups(cx.r, cx).to_diagram()\
        == compact.Cup(cx.r, cx)
    assert compact.CMap.caps(cx, cx.r).to_diagram()\
        == compact.Cap(cx, cx.r)


def test_only_to_diagram_needs_the_structure():
    from discopy import cmap, monoidal, pivotal, traced

    x = monoidal.Ty("x")
    f = monoidal.Box("f", x, x)
    cycle = monoidal.CMap(x, x, (f, ), (3, 2, 1, 0))
    assert not cycle.is_acyclic
    with raises(AxiomError, match="has no traces"):
        cycle.to_diagram()

    tx = traced.Ty("x")
    feedback = traced.CMap(
        tx, tx, (traced.Box("f", tx, tx), ), (3, 2, 1, 0))
    assert not feedback.is_acyclic

    class Planar(symmetric.Diagram):
        """ A category with adjoint types but no cups or caps. """
        ob = pivotal.Ty

    p = pivotal.Ty("p")
    cup = cmap.CMap[Planar](p @ p.r, pivotal.Ty(), (), (1, 0))
    assert not cup.is_monogamous
    with raises(AxiomError, match="has no cups or caps"):
        cup.to_diagram()


def test_explicit_trace_on_a_subclass():
    from discopy import cmap, symmetric as sym
    from discopy.utils import factory

    @factory
    class Recipe(sym.Diagram):
        """ A user subclass of a symmetric category. """

    class Step(sym.Box, Recipe):
        """ A box in a recipe. """

    class Reduce(sym.Trace, Step):
        """ A recipe knows how to feed one of its outputs back. """

    Recipe.trace_factory = Reduce
    x = sym.Ty("x")
    f = Step("f", x, x)

    traced = cmap.CMap[Recipe].from_box(f).trace()
    assert traced.to_diagram() == Reduce(f, False)
    assert f.to_hypergraph().trace().to_diagram() == Reduce(f, False)

    Recipe.trace_factory = sym.Trace
    with raises(TypeError, match="Expected .*Recipe, got symmetric.Trace"):
        cmap.CMap[Recipe].from_box(f).trace().to_diagram()


def test_composed_snakes_are_reordered_on_downgrade():
    x = compact.Ty("x")
    f, g = compact.Box("f", x, x), compact.Box("g", x, x)
    snakes = f.transpose(left=True) >> g.transpose(left=True)

    cm = snakes.to_map()
    assert not cm.is_monogamous and cm.is_acyclic
    assert not cm.is_topologically_ordered and not cm.is_causal
    assert cm.topological_order().boxes == (g, f)
    once = cm.to_diagram().to_map()
    assert once.to_diagram().to_map() == once
    assert snakes.to_hypergraph().to_diagram().to_map().to_hypergraph()\
        == once.to_hypergraph()


def test_is_causal_is_local():
    from discopy import traced

    x = compact.Ty("x")
    f, g = compact.Box("f", x, x), compact.Box("g", x, x)
    y = traced.Ty("x")
    h = traced.Box("h", y, y)
    maps = [
        compact.CMap.id(x),
        f.to_map() >> g.to_map(),
        (f.to_map() @ g.to_map()).interchange(0, 1),
        compact.CMap.cups(x, x.r),
        compact.CMap.caps(x.r, x),
        (f.transpose(left=True) >> g.transpose(left=True)).to_map(),
        h.to_map().trace(),
        traced.CMap(traced.Ty(), traced.Ty(), (), [], loops=(y, )),
        compact.CMap(x, x, (f, g), [3, 4, 5, 0, 1, 2], check=False)]
    for cm in maps:
        assert cm.is_causal == (
            cm.is_monogamous and cm.is_acyclic
            and cm.is_topologically_ordered)


def test_unordered_boxes_are_reordered_on_downgrade():
    from discopy import symmetric as sym

    x = sym.Ty("x")
    f, g = sym.Box("f", x, x), sym.Box("g", x, x)
    assert (f >> g).to_map().edges == Permutation([1, 0, 3, 2, 5, 4], 6)
    assert sym.CMap(x, x, (f, g), [1, 0, 3, 2, 5, 4]).is_causal
    unordered = sym.CMap(x, x, (f, g), [3, 4, 5, 0, 1, 2])
    assert unordered.is_acyclic and not unordered.is_topologically_ordered
    assert unordered.to_diagram() == g >> f


@pytest.mark.parametrize(
    "module",
    [
        compact,
        closed,
        biclosed,
    ]
)
def test_curry_uncurry_roundtrip(module):
    x, y, z = map(module.Ty, "xyz")
    f = module.Box("f", x @ y, z)
    cmap = f.to_map()

    assert cmap.curry(n=0).uncurry(n=0) == cmap
    with raises(ValueError):
        cmap.curry(n=3)
    with raises(ValueError):
        cmap.uncurry(n=2)

    if module is compact:
        assert cmap.curry().uncurry() == cmap
        assert cmap.curry(left=True).uncurry(left=True) == cmap
        assert cmap.curry(n=2, left=True).uncurry(n=2, left=True) == cmap
        return

    right = cmap.curry()
    assert right.dom == y
    assert right.cod == x >> z
    assert right.boxes == (module.Diagram.curry_factory(f, 1, False), )
    assert f.curry().to_map() == right

    left = cmap.curry(left=True)
    assert left.dom == x
    assert left.cod == z << y
    assert left.boxes == (module.Diagram.curry_factory(f, 1, True), )
    assert f.curry(left=True).to_map() == left

    h = module.Box("h", y, x >> z)
    uncurried = h.to_map().uncurry()
    assert uncurried.dom == x @ y
    assert uncurried.cod == z
    assert uncurried.boxes == (
        h, module.Diagram.eval_factory(x >> z, left=False))
    assert h.uncurry().to_map() == uncurried

    w = module.Ty("w")
    k = module.Box("k", x @ y @ z, w)
    right_two = k.to_map().curry(n=2).uncurry(n=2)
    assert right_two.dom == x @ y @ z
    assert right_two.cod == w
    assert right_two.boxes == (
        module.Diagram.curry_factory(k, 2, False),
        module.Diagram.eval_factory(x @ y >> w, left=False))

    left_two = k.to_map().curry(n=2, left=True).uncurry(
        n=2, left=True)
    assert left_two.dom == x @ y @ z
    assert left_two.cod == w
    assert left_two.boxes == (
        module.Diagram.curry_factory(k, 2, True),
        module.Diagram.eval_factory(w << y @ z, left=True))

    right_nested = k.to_map().curry().curry().uncurry(n=2)
    assert right_nested.dom == x @ y @ z
    assert right_nested.cod == w

    left_nested = k.to_map().curry(left=True).curry(
        left=True).uncurry(n=2, left=True)
    assert left_nested.dom == x @ y @ z
    assert left_nested.cod == w

    with raises(ValueError):
        k.to_map().curry(n=2).uncurry()


def test_trace():
    from discopy.compact import Ty, Box, CMap as M

    x, y = map(Ty, "xy")
    assert M.id(x).trace().loops == (x, )
    assert M.id(x).trace(left=True).loops == (x, )
    assert M.swap(x, x).trace() == M.id(x)

    f = M.from_box(Box("f", x @ y, x @ y))
    right_trace = f.trace()
    assert right_trace.dom == x
    assert right_trace.cod == x
    assert right_trace.boxes == f.boxes

    left_trace = f.trace(left=True)
    assert left_trace.dom == y
    assert left_trace.cod == y
    assert left_trace.boxes == f.boxes

    closed_component = M.from_box(Box("h", x, x)).trace()
    assert closed_component.dom == Ty()
    assert closed_component.cod == Ty()
    assert len(closed_component.boxes) == 1
    assert closed_component.edges == (1, 0)
    assert closed_component.loops == ()
    assert closed_component.boundary_cycle == ()
    assert closed_component.n_vertices == 1
    assert closed_component.euler_characteristic == 2
    assert closed_component.is_planar


def test_make_causal_cuts_every_backward_wire_at_once():
    from discopy import traced
    x = traced.Ty("x")
    f, g = traced.Box("f", x @ x, x @ x), traced.Box("g", x, x)
    cmap = (f.to_map() >> g.to_map() @ x).trace(2)
    assert not cmap.is_acyclic
    assert cmap.make_causal().boxes == (
        traced.Trace(traced.Trace(f >> g @ x)), )
    assert cmap.to_diagram().to_map() == cmap


def test_scalar_box():
    from discopy.compact import Ty, Box, CMap as M

    s = Box("s", Ty(), Ty())
    cm = M.from_box(s)
    assert cm.edges == ()
    assert cm.orientation == ()
    assert cm.faces == ()
    assert cm.euler_characteristic == 2
    assert cm.is_scalar
    assert cm.is_planar
    assert cm.to_hypergraph() == s.to_hypergraph()


def test_zipping_cups_and_caps():
    """
    │ ╭─╮ ╭─╮ ╭─╮ ╭─╮    │
    │ │ │ │ │ │ │ │ │  = │
    ╰─╯ ╰─╯ ╰─╯ ╰─╯ │    │
    """

    from discopy.compact import Ty, Diagram as D, CMap as M

    x, y = map(Ty, 'xy')

    def zipping_expr(c, z):
        id, cup, cap = c.id(z), c.cups(z, z.r), c.caps(z.r, z)
        return id @ cap @ cap @ cap @ cap >> cup @ cup @ cup @ cup @ id

    assert zipping_expr(D, x).to_map() == zipping_expr(M, x) == M.id(x)
    assert zipping_expr(D, x @ y).to_map()\
        == zipping_expr(M, x @ y) == M.id(x @ y)


def test_scalar_is_not_eliminated():
    from discopy.compact import Ty, Diagram as D, CMap as M

    x = Ty("x")
    scalar_map = M.caps(x.r, x) >> M.cups(x.r, x)
    scalar_dgm = D.caps(x.r, x) >> D.cups(x.r, x)

    assert scalar_map != M.id()
    assert scalar_map.loops == (x,)
    assert scalar_map.euler_characteristic == 0
    assert scalar_map.is_scalar
    assert scalar_map.is_planar
    assert (D.caps(x.r, x) >> D.cups(x.r, x)).to_map() == scalar_map
    assert scalar_map.to_hypergraph() == scalar_dgm.to_hypergraph()


def test_connected_components_of_loops():
    from discopy.compact import Ty, CMap as M

    x, y = Ty("x"), Ty("y")
    loops = (M.caps(x.r, x) >> M.cups(x.r, x))\
        @ (M.caps(y.r, y) >> M.cups(y.r, y))
    assert loops.loops == (x, y)
    components = loops.connected_components
    assert len(components) == 2
    assert tuple(c.loops for c in components) == ((x,), (y,))


def test_hypergraph_to_map():
    from discopy import compact, frobenius

    x, y = map(compact.Ty, "xy")
    f = compact.Box("f", x, y).to_hypergraph()
    assert f.to_map().to_hypergraph() == f

    fx = frobenius.Ty("x")
    assert frobenius.Hypergraph.spiders(1, 2, fx).to_map()\
        == frobenius.CMap.spiders(1, 2, fx)


def test_then():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z, w = map(Ty, "xyzw")
    f, g, h = [
        M.from_box(box) for box in [
            Box("f", x, y), Box("g", y, z), Box("h", z, w)]
    ]
    assert ((f >> g) >> h) == (f >> (g >> h))
    assert (f >> M.id(y)) == f
    assert (M.id(x) >> f) == f
    assert (f >> g).to_hypergraph() == f.to_hypergraph() >> g.to_hypergraph()
    with raises(AxiomError):
        f >> f


def test_tensor():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z = map(Ty, "xyz")
    f = M.from_box(Box("f", x, y))
    g = M.from_box(Box("g", y, z))
    assert (f @ g).to_hypergraph() == f.to_hypergraph() @ g.to_hypergraph()
    assert (f @ M.id()) == f
    assert (M.id() @ f) == f


@pytest.mark.parametrize(
    "module",
    [
        symmetric,
        compact,
        closed,
    ]
)
def test_interchange(module):
    Ty, Box, M = module.Ty, module.Box, module.CMap

    # interchange of independent boxes
    x, y, z, w, a, b = map(Ty, "xyzwab")
    f, g, h = Box("f", x, y), Box("g", z, w), Box("h", a, b)
    cm = M.from_box(f) @ M.from_box(g) @ M.from_box(h)
    swapped = cm.interchange(0, 2)
    assert swapped.boxes == (h, g, f)
    assert swapped.dom == cm.dom
    assert swapped.cod == cm.cod
    assert swapped.edges == Permutation.from_transpositions(
        [(0, 7), (1, 5), (2, 3), (4, 11), (6, 10), (8, 9)],
        12,
    )
    assert swapped != cm
    assert swapped.interchange(2, 0) == cm
    with raises(IndexError):
        cm.interchange(0, 3)

    f, g = Box("f", x, y), Box("t", y, z)
    cm = M.from_box(f) >> M.from_box(g)
    assert cm.is_causal
    unordered = cm.interchange(0, 1)
    assert unordered.boxes == (g, f)
    assert not unordered.is_topologically_ordered
    assert unordered.topological_order() == cm


def test_plug_input():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z = map(Ty, "xyz")
    direct = M.id(x).plug_input(0, Box("lambda", x, y @ x), y)
    assert direct.dom == Ty()
    assert direct.cod == y
    assert direct.orientation == Permutation.from_cycles(
        [(0, 1, 2), (3,)], 4)

    f = M.from_box(Box("f", z, x))
    indirect = f.plug_input(0, Box("lambda", x, y @ z), y)
    assert indirect.dom == Ty()
    assert indirect.cod == y
    assert len(indirect.boxes) == 2

    right_root = M.id(x).plug_input(
        0, Box("lambda", x, x @ y), y, root_index=1)
    assert right_root.dom == Ty()
    assert right_root.cod == y
    with raises(ValueError):
        f.plug_input(-1, Box("lambda", x, y @ z), y)
    with raises(ValueError):
        f.plug_input(0, Box("lambda", x, y @ z), y, root_index=2)
    with raises(ValueError):
        f.plug_input(0, Box("bad", Ty(), y @ z), y)


def test_tensor_then():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z, a, b = map(Ty, "xyzab")
    f1 = M.from_box(Box("f1", x, y))
    f2 = M.from_box(Box("f2", y, z))
    g = M.from_box(Box("g", a, b))
    assert ((f1 >> f2) @ g).to_hypergraph() == (
        f1.to_hypergraph() >> f2.to_hypergraph()
    ) @ g.to_hypergraph()


def test_then_tensor():
    from discopy.compact import Ty, Box, CMap as M
    x1, x2, y1, y2, z = map(Ty, ["x1", "x2", "y1", "y2", "z"])
    f1 = M.from_box(Box("f1", x1, y1))
    f2 = M.from_box(Box("f2", x2, y2))
    g = M.from_box(Box("g", y1 @ y2, z))
    assert ((f1 @ f2) >> g).to_hypergraph() == (
        f1.to_hypergraph() @ f2.to_hypergraph()
    ) >> g.to_hypergraph()


def test_euler_characteristic():
    from discopy import closed, compact

    x, y = map(closed.Ty, "xy")
    assert closed.CMap.id().is_planar
    wire = closed.CMap.id(x)
    box = closed.CMap.from_box(closed.Box("f", x, y))
    assert wire.faces == Permutation.from_cycles([(0,), (1,)], 2)
    assert wire.n_vertices == 1
    assert wire.n_edges == 1
    assert wire.n_faces == 2
    assert wire.euler_characteristic == 2
    assert wire.is_planar
    assert box.faces == Permutation.from_cycles([(0, 2), (1, 3)], 4)
    assert box.n_vertices == 2
    assert box.n_edges == 2
    assert box.n_faces == 2
    assert box.euler_characteristic == 2
    assert not box.is_scalar
    assert box.is_planar

    cx, cy = map(compact.Ty, "xy")
    cbox = compact.Box("f", cx, cy).to_map()
    scalar = cbox.caps(cx.r, cx) >> cbox.cups(cx.r, cx)
    assert scalar.euler_characteristic == 0
    assert scalar.is_scalar
    assert scalar.is_planar
    assert (cbox @ scalar).is_planar
    assert not (cbox @ scalar).is_scalar
    with raises(ValueError):
        (cbox @ scalar).euler_characteristic
    components = (cbox @ scalar).connected_components
    assert components == [cbox, scalar]

    s = compact.Box("s", compact.Ty(), compact.Ty()).to_map()
    t = compact.Box("t", compact.Ty(), compact.Ty()).to_map()
    assert (s @ t).connected_components == [s, t]
    assert compact.CMap.id().connected_components == [compact.CMap.id()]


def test_draw_plain_path(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("needs the graphviz dot binary")
    f = compact.Box("f", compact.Ty("x"), compact.Ty("y")).to_map()
    for fmt in ("dot", "svg"):
        path = tmp_path / f"f.{fmt}"
        for _ in range(2):  # A plain path saves, overwriting silently.
            f.draw(path=path, show=False)
        assert path.exists()


def test_boxes_with_no_domain_decode_at_the_right():
    x, y = map(compact.Ty, "xy")
    f = compact.Box("f", x, y)
    snake = f.transpose(left=True)
    assert snake.to_map().to_diagram() == snake
    assert "Swap" not in str(snake.to_map().to_diagram())


def test_curry_is_wiring_only_when_the_category_is_rigid():
    from discopy import pivotal, rigid

    for module in (biclosed, closed):
        x, y, z = map(module.Ty, "xyz")
        f = module.Box("f", x @ y, z)
        curried = f.to_map().curry()
        assert curried.boxes == (module.Diagram.curry_factory(f, 1, False), )
        assert curried.uncurry() != f.to_map()

    for module in (compact, rigid, pivotal):
        x, y, z = map(module.Ty, "xyz")
        f = module.Box("f", x @ y, z)
        curried = f.to_map().curry(left=True)
        assert curried.boxes == (f, ) and not curried.is_monogamous
        assert curried.uncurry(left=True) == f.to_map()

    x, y, z = map(rigid.Ty, "xyz")
    f = rigid.Box("f", x @ y, z)
    assert f.to_map().curry(left=True).make_monogamous().boxes == (
        rigid.Cap(y, y.l), f)
    with raises(AxiomError, match="adjoint"):
        f.to_map().curry(left=False).make_monogamous()

    x, y, z = map(pivotal.Ty, "xyz")
    f = pivotal.Box("f", x @ y, z)
    assert f.to_map().curry(left=False).make_monogamous().boxes == (
        pivotal.Cap(x, x.r), f)


def test_map_and_hypergraph_normalise_the_same_way():
    from discopy import frobenius, monoidal, rigid, traced

    x, y = map(compact.Ty, "xy")
    f = compact.Box("f", x, y)
    g, h = compact.Box("g", x, x), compact.Box("h", x, x)
    sx = symmetric.Ty("x")
    sf, sg = symmetric.Box("f", sx, sx), symmetric.Box("g", sx, sx)
    mx = monoidal.Ty("x")
    maps = [
        monoidal.CMap(mx, mx, (monoidal.Box("f", mx, mx), ), (3, 2, 1, 0)),
        (sf.to_map() >> sg.to_map()).interchange(0, 1),
        symmetric.Box("f", sx @ sx, sx @ sx).to_map().trace(),
        f.transpose(left=True).to_map(),
        (g.transpose(left=True) >> h.transpose(left=True)).to_map(),
        compact.CMap.caps(x.r, x) >> compact.CMap.cups(x.r, x),
        rigid.Box("f", rigid.Ty("x"), rigid.Ty("y")).to_map(),
        frobenius.Diagram.spiders(2, 1, frobenius.Ty("x")).to_map(),
        traced.Box("f", traced.Ty("x"), traced.Ty("x")).to_map().trace()]
    for cm in maps:
        hyp = cm.to_hypergraph()
        assert cm.is_monogamous == hyp.is_monogamous
        assert cm.is_acyclic == hyp.is_acyclic
        assert cm.is_topologically_ordered == hyp.is_topologically_ordered
        if cm.is_monogamous:
            continue
        assert cm.make_monogamous().boxes == hyp.make_monogamous().boxes
