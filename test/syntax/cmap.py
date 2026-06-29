from pytest import raises

from discopy.python.finset import Permutation
from discopy.utils import AxiomError


def to_hypergraph(cmap):
    return cmap.category.hypergraph_factory.from_map(cmap)


def test_port_side_and_direction():
    from discopy.compact import Ty, CMap as M
    x = Ty("x")
    ports = M.id(x).ports
    assert ports[0].side == "up"
    assert ports[1].side == "down"
    assert ports[0].direction == "up"
    assert ports[1].direction == "down"
    adjoint_ports = M.id(x.r).ports
    assert adjoint_ports[0].side == "up"
    assert adjoint_ports[1].side == "down"
    assert adjoint_ports[0].direction == "down"
    assert adjoint_ports[1].direction == "up"


def test_default_compact_setting():
    from discopy.compact import Ty, Box, CMap as M
    x, y = map(Ty, "xy")
    f = Box("f", x, y)
    cm = M.from_box(f)
    assert isinstance(f, M.category)
    assert to_hypergraph(cm).category == M.category


def test_M_init():
    from discopy.compact import Ty, Box, CMap as M
    x, y, z = map(Ty, "xyz")
    f = Box("f", x @ y, z)
    valid = M.from_box(f)
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
    with raises(ValueError):
        M(f.dom, f.cod, (f,), valid.edges, offsets=(None, None))


def test_repr_eq_and_hash():
    from discopy.compact import Ty, Box, CMap as M

    x, y = map(Ty, "xy")
    cm = M.from_box(Box("f", x, y))
    assert "ports=" in repr(cm)
    assert cm == M.from_box(Box("f", x, y))
    assert cm != object()
    assert hash(cm) == hash(M.from_box(Box("f", x, y)))


def test_id_and_tensor():
    from discopy.compact import Ty, CMap as M, Hypergraph as H
    x, y = map(Ty, "xy")
    assert M.id(x).edges == (1, 0)
    assert M.id(x).orientation == (1, 0)
    assert M.id(x).faces == (0, 1)
    assert M.id().tensor() == M.id()
    assert M.id(x).tensor(M.id(y)) == M.id(x) @ M.id(y)
    assert to_hypergraph(M.id(x) @ M.id(y)) == H.id(x @ y)


def test_from_box_and_to_hypergraph():
    from discopy.compact import Ty, Box, CMap as M
    x, y, z = map(Ty, "xyz")
    f = Box("f", x, y)
    cm = M.from_box(f)
    assert cm.edges == (1, 0, 3, 2)
    assert cm.orientation == (3, 2, 1, 0)
    assert cm.faces == (2, 3, 0, 1)
    assert to_hypergraph(cm) == f.to_hypergraph()

    multi_input = M.from_box(Box("g", x @ y, z))
    assert multi_input.orientation == Permutation.from_cycles(
        [(1, 0, 5), (2, 3, 4)], 6)


def test_eliminate_swaps():
    from discopy.compact import Ty, Id, Box

    x, y, w, z = map(Ty, "xyzw")

    diagram = Id(x @ y).swap(x, y).swap(y, x)
    assert diagram == diagram.to_map().to_diagram().normal_form()

    diagram = Id(x @ y @ w @ z).swap(x @ y, w @ z).swap(w @ z, x @ y).normal_form()
    assert diagram == diagram.to_map().to_diagram().normal_form()

    f, g = Box("f", x, z), Box("g", y, w)

    diagram = Id(x @ y).swap(x, y) >> g @ x >> Id(w @ x).swap(w, x) >> f @ w
    assert diagram == diagram.to_map().to_diagram().normal_form()
    assert diagram.to_map() == diagram.to_hypergraph().to_diagram().to_map()


def test_to_diagram_preserves_offsets():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z = map(Ty, "xyz")
    delayed = Box("h", x @ y, z)
    cmap = M(delayed.dom, delayed.cod, (delayed, ),
             M.from_box(delayed).edges, offsets=(2, ))
    assert cmap.to_diagram().to_map() == cmap


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
        monoidal.CMap(x @ x, x @ x, (), (3, 2, 1, 0))
    assert symmetric.CMap(x @ x, x @ x, (), (3, 2, 1, 0))\
        == symmetric.CMap.swap(x, x)

    x, y, z = map(monoidal.Ty, "xyz")
    f = monoidal.Box("f", x @ y, z)
    with raises(AxiomError):
        monoidal.CMap(y @ x, z, (f, ), (3, 2, 1, 0, 5, 4))


def test_diagram_to_map_structure_and_errors():
    from discopy import (
        balanced,
        braided,
        closed,
        compact,
        frobenius,
        markov,
        monoidal,
        symmetric,
        traced,
    )
    from discopy.cmap import Port, PortKind

    mx, my = map(monoidal.Ty, "xy")
    f = monoidal.Box("f", mx, my)
    assert monoidal.CMap.require_planar is True
    assert f.to_map() == monoidal.CMap.from_box(f)

    bx, by = map(braided.Ty, "xy")
    braid = braided.Braid(bx, by)
    assert monoidal.CMap.from_diagram(braid).boxes == (braid, )

    sx, sy = map(symmetric.Ty, "xy")
    assert symmetric.CMap.require_planar is False
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

    bx = balanced.Ty("x")
    twist = balanced.Twist(bx)
    assert traced.CMap.from_diagram(twist).boxes == (twist, )
    assert twist.to_map().boxes == (twist, )

    cx, cy = map(closed.Ty, "xy")
    ev = closed.Eval(cy << cx)
    assert ev.to_map() == closed.CMap.ev(cy, cx, left=False)
    assert ev.to_map().boxes == (ev, )

    mx = markov.Ty("x")
    copy = markov.Copy(mx, 2)
    assert copy.to_map() == markov.CMap.copy(mx, 2)

    fx = frobenius.Ty("x")
    spider = frobenius.Spider(1, 2, fx)
    assert markov.CMap.from_diagram(spider).boxes == (spider, )
    assert spider.to_map() == frobenius.CMap.spiders(1, 2, fx)

    x, y = map(compact.Ty, "xy")
    assert to_hypergraph(compact.CMap.swap(x, y)) == compact.CMap.category.swap(
        x, y).to_hypergraph()
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
    with raises(AxiomError):
        monoidal.CMap.cups(x, x)
    with raises(AxiomError):
        monoidal.CMap.caps(x, x)
    assert monoidal.CMap.id(x).edges == (1, 0)
    f = monoidal.Box("f", x, x)
    g = monoidal.Box("g", x, x)
    with raises(AxiomError):
        monoidal.CMap(monoidal.Ty(), monoidal.Ty(), (f, g), (3, 2, 1, 0))

    x = traced.Ty("x")
    with raises(AxiomError):
        traced.CMap.cups(x, x)
    with raises(AxiomError):
        traced.CMap.caps(x, x)

    x, y = map(closed.Ty, "xy")
    assert closed.CMap.ev(y, x).boxes == (
        closed.CMap.category.ev(y, x), )

    x = markov.Ty("x")
    assert markov.CMap.copy(x, 2).boxes == (markov.CMap.category.copy(x, 2), )
    assert markov.CMap.merge(x, 2).boxes == (markov.CMap.category.merge(x, 2), )
    assert markov.CMap.discard(x).boxes == (markov.CMap.category.copy(x, 0), )

    x = frobenius.Ty("x")
    assert frobenius.CMap.spiders(1, 2, x).boxes == (
        frobenius.Diagram.spiders(1, 2, x), )
    assert frobenius.Diagram.map_factory is frobenius.CMap


def test_trace():
    from discopy.compact import Ty, Box, CMap as M

    x, y = map(Ty, "xy")
    assert M.id(x).trace().scalars == (x, )
    assert M.id(x).trace(left=True).scalars == (x, )
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

    scalar_component = M.from_box(Box("h", x, x)).trace()
    assert scalar_component.dom == Ty()
    assert scalar_component.cod == Ty()
    assert len(scalar_component.boxes) == 1
    assert scalar_component.scalars == ()
    assert scalar_component.boundary_cycle == ()
    assert scalar_component.n_vertices == 1
    assert scalar_component.euler_characteristic == 2
    assert scalar_component.is_planar


def test_curry_uncurry_roundtrip():
    from discopy.compact import Ty, Box

    x, y, z = map(Ty, "xyz")
    cmap = Box("f", x @ y, z).to_map()
    assert cmap.curry().uncurry() == cmap
    assert cmap.curry(left=True).uncurry(left=True) == cmap
    assert cmap.curry(n=0).uncurry(n=0) == cmap
    assert cmap.curry(n=2, left=True).uncurry(n=2, left=True) == cmap
    with raises(ValueError):
        cmap.curry(n=3)
    with raises(ValueError):
        cmap.uncurry(n=2)


def test_scalar_box():
    from discopy.compact import Ty, Box, CMap as M

    s = Box("s", Ty(), Ty())
    cm = M.from_box(s)
    assert cm.edges == ()
    assert cm.orientation == ()
    assert cm.faces == ()
    assert cm.euler_characteristic == 1
    assert to_hypergraph(cm) == s.to_hypergraph()


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
    assert zipping_expr(D, x @ y).to_map() == zipping_expr(M, x @ y) == M.id(x @ y)


def test_scalar_is_not_eliminated():
    from discopy.compact import Ty, Diagram as D, CMap as M

    x = Ty("x")
    scalar = M.caps(x.r, x) >> M.cups(x.r, x)

    assert scalar != M.id()
    assert scalar.scalars == (x,)
    assert scalar.euler_characteristic == 1
    assert (D.caps(x.r, x) >> D.cups(x.r, x)).to_map() == scalar
    assert to_hypergraph(scalar).to_map() == scalar
    dot = scalar.to_dot()
    assert "scalar0" in dot
    assert 'scalar0 -- scalar0 [len="0.85", label="x"];' in dot


def test_hypergraph_to_map():
    from discopy.compact import Ty, Box

    x, y = map(Ty, "xy")
    f = Box("f", x, y).to_hypergraph()
    assert to_hypergraph(f.to_map()) == f

    from discopy.frobenius import Ty as FTy, CMap as FM, Hypergraph as FH

    fx = FTy("x")
    assert FH.spiders(1, 2, fx).to_map() == FM.spiders(1, 2, fx)


def test_then():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z, w = map(Ty, "xyzw")
    f, g, h = [
        M.from_box(box) for box in [Box("f", x, y), Box("g", y, z), Box("h", z, w)]
    ]
    assert ((f >> g) >> h) == (f >> (g >> h))
    assert (f >> M.id(y)) == f
    assert (M.id(x) >> f) == f
    assert to_hypergraph(f >> g) == to_hypergraph(f) >> to_hypergraph(g)
    with raises(AxiomError):
        f >> f


def test_tensor():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z = map(Ty, "xyz")
    f = M.from_box(Box("f", x, y))
    g = M.from_box(Box("g", y, z))
    assert to_hypergraph(f @ g) == to_hypergraph(f) @ to_hypergraph(g)
    assert (f @ M.id()) == f
    assert (M.id() @ f) == f


def test_interchange():
    from discopy.compact import Ty, Box, CMap as M

    x, y, z, w, a, b = map(Ty, "xyzwab")
    f, g, h = Box("f", x, y), Box("g", z, w), Box("h", a, b)
    cm = M.from_box(f) @ M.from_box(g) @ M.from_box(h)
    swapped = cm.interchange(0, 2)
    assert swapped.boxes == (h, g, f)
    assert swapped.dom == cm.dom
    assert swapped.cod == cm.cod
    assert swapped.edges == (7, 5, 3, 2, 11, 1, 10, 0, 9, 8, 6, 4)
    assert swapped != cm
    assert swapped.interchange(2, 0) == cm
    with raises(IndexError):
        cm.interchange(0, 3)

    scalar_f, scalar_g = Box("s", Ty(), Ty()), Box("t", Ty(), Ty())
    scalar = M.from_box(scalar_f) @ M.from_box(scalar_g)
    assert scalar.interchange(0, 1).boxes == (scalar_g, scalar_f)


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
    assert to_hypergraph((f1 >> f2) @ g) == (
        to_hypergraph(f1) >> to_hypergraph(f2)
    ) @ to_hypergraph(g)


def test_then_tensor():
    from discopy.compact import Ty, Box, CMap as M
    x1, x2, y1, y2, z = map(Ty, ["x1", "x2", "y1", "y2", "z"])
    f1 = M.from_box(Box("f1", x1, y1))
    f2 = M.from_box(Box("f2", x2, y2))
    g = M.from_box(Box("g", y1 @ y2, z))
    assert to_hypergraph((f1 @ f2) >> g) == (
        to_hypergraph(f1) @ to_hypergraph(f2)
    ) >> to_hypergraph(g)


def test_euler_characteristic():
    from discopy.closed import Ty, Box, CMap as M
    from discopy.compact import Ty as CTy, Box as CBox
    x, y = map(Ty, "xy")
    assert M.id().is_planar
    wire = M.id(x)
    box = M.from_box(Box("f", x, y))
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
    assert box.is_planar

    cx, cy = map(CTy, "xy")
    cbox = CBox("f", cx, cy).to_map()
    scalar = cbox.caps(cx.r, cx) >> cbox.cups(cx.r, cx)
    assert scalar.euler_characteristic == 1
    assert scalar.is_planar
    assert (cbox @ scalar).euler_characteristic == 3
    assert not (cbox @ scalar).is_planar
