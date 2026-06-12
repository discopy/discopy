import shutil

from pytest import raises

from discopy.combinatorial_map import (
    Permutation,
    port_side,
)
from discopy.utils import AxiomError


def test_cycles():
    assert Permutation((1, 0, 3, 2)).cycles() == ((0, 1), (2, 3))
    assert Permutation.from_cycles([(0, 1), (2, 3)], 4) == (1, 0, 3, 2)
    assert Permutation((1, 0)).is_fixpoint_free_involution()
    assert not Permutation((0,)).is_fixpoint_free_involution()
    assert Permutation.identity(2) == (0, 1)
    assert Permutation((1, 0)).compose((1, 0)) == (0, 1)
    assert Permutation((1, 0)).tensor((1, 0)) == (1, 0, 3, 2)


def test_port_side():
    from discopy.compact import Ty, CombinatorialMap as M
    x = Ty("x")
    ports = M.id(x).ports
    assert port_side(ports[0]) == "left"
    assert port_side(ports[1]) == "right"


def test_default_compact_setting():
    from discopy.compact import Ty, Box, CombinatorialMap as M
    x, y = map(Ty, "xy")
    f = Box("f", x, y)
    cmap = M.from_box(f)
    assert isinstance(f, M.category)
    assert cmap.to_hypergraph().category == M.category


def test_M_init():
    from discopy.compact import Ty, Box, CombinatorialMap as M
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
    with raises(ValueError):
        g = Box("g", x, x)
        M(g.dom, g.cod, (g,), M.from_box(g).edge, (1, 0, 2, 3))
    assert M(f.dom, f.cod, (f,), valid.edge, (0, 1, 4, 2, 3, 5)).node_cycles == (
        (2, 4, 3),
    )
    with raises(ValueError):
        M(f.dom, f.cod, (f,), valid.edge, tuple(range(valid.n_ports)))
    with raises(AxiomError):
        M(x, y, (), (1, 0))


def test_id_and_tensor():
    from discopy.compact import Ty, CombinatorialMap as M, Hypergraph as H
    x, y = map(Ty, "xy")
    assert M.id(x).edge == (1, 0)
    assert M.id(x).node == (0, 1)
    assert M.id().tensor() == M.id()
    assert M.id(x).tensor(M.id(y)) == M.id(x) @ M.id(y)
    assert (M.id(x) @ M.id(y)).to_hypergraph() == H.id(x @ y)


def test_from_box_and_to_hypergraph():
    from discopy.compact import Ty, Box, CombinatorialMap as M
    x, y = map(Ty, "xy")
    f = Box("f", x, y)
    cmap = M.from_box(f)
    assert cmap.edge == (1, 0, 3, 2)
    assert cmap.node == (0, 2, 1, 3)
    assert cmap.to_hypergraph() == f.to_hypergraph()


def test_scalar_box():
    from discopy.compact import Ty, Box, CombinatorialMap as M

    s = Box("s", Ty(), Ty())
    cmap = M.from_box(s)
    assert cmap.edge == ()
    assert cmap.node == ()
    assert cmap.euler_characteristic == 1
    assert cmap.to_hypergraph() == s.to_hypergraph()


def test_from_hypergraph():
    from discopy.compact import Ty, Box, CombinatorialMap as M, Hypergraph as H

    x, y = map(Ty, "xy")
    f = Box("f", x, y).to_hypergraph()
    assert M.from_hypergraph(f).to_hypergraph() == f
    with raises(ValueError):
        M.from_hypergraph(H.spiders(1, 2, x))


def test_then():
    from discopy.compact import Ty, Box, CombinatorialMap as M

    x, y, z, w = map(Ty, "xyzw")
    f, g, h = [
        M.from_box(box) for box in [Box("f", x, y), Box("g", y, z), Box("h", z, w)]
    ]
    assert ((f >> g) >> h) == (f >> (g >> h))
    assert (f >> M.id(y)) == f
    assert (M.id(x) >> f) == f
    assert (f >> g).to_hypergraph() == f.to_hypergraph() >> g.to_hypergraph()
    with raises(AxiomError):
        f >> f


def test_tensor():
    from discopy.compact import Ty, Box, CombinatorialMap as M

    x, y, z = map(Ty, "xyz")
    f = M.from_box(Box("f", x, y))
    g = M.from_box(Box("g", y, z))
    assert (f @ g).to_hypergraph() == f.to_hypergraph() @ g.to_hypergraph()
    assert (f @ M.id()) == f
    assert (M.id() @ f) == f


def test_tensor_then():
    from discopy.compact import Ty, Box, CombinatorialMap as M

    x, y, z, a, b = map(Ty, "xyzab")
    f1 = M.from_box(Box("f1", x, y))
    f2 = M.from_box(Box("f2", y, z))
    g = M.from_box(Box("g", a, b))
    assert ((f1 >> f2) @ g).to_hypergraph() == (
        f1.to_hypergraph() >> f2.to_hypergraph()
    ) @ g.to_hypergraph()


def test_then_tensor():
    from discopy.compact import Ty, Box, CombinatorialMap as M
    x1, x2, y1, y2, z = map(Ty, ["x1", "x2", "y1", "y2", "z"])
    f1 = M.from_box(Box("f1", x1, y1))
    f2 = M.from_box(Box("f2", x2, y2))
    g = M.from_box(Box("g", y1 @ y2, z))
    assert ((f1 @ f2) >> g).to_hypergraph() == (
        f1.to_hypergraph() @ f2.to_hypergraph()
    ) >> g.to_hypergraph()


def test_euler_characteristic():
    from discopy.closed import Ty, Box, CombinatorialMap as M
    x, y = map(Ty, "xy")
    wire = M.id(x)
    box = M.from_box(Box("f", x, y))
    scalar = M.from_box(Box("s", Ty(), Ty()))
    assert wire.face_cycles == ((0, 1),)
    assert wire.euler_characteristic == 0
    assert box.face_cycles == ((0, 2, 3, 1),)
    assert box.euler_characteristic == 0
    assert (box @ scalar).euler_characteristic == 1


def assert_trivalent_map(cmap, dom, cod, vertices):
    assert cmap.dom == dom
    assert cmap.cod == cod
    assert len(cmap.cod) == 1
    assert len(cmap.boxes) == vertices
    assert all(len(cycle) == 3 for cycle in cmap.node_cycles)
    assert cmap.ports[-1].kind == "output"
    assert cmap.node[-1] == len(cmap.ports) - 1
    assert cmap.edge[-1] != len(cmap.ports) - 1


def assert_freevars_as_domain(term, cmap):
    from discopy.closed import Ty
    dom = Ty()
    for variable in term.freevars:
        dom = dom @ variable.cod
    assert cmap.dom == dom


def assert_round_trips_through_term(cmap, input_names=None):
    term = cmap.to_term(input_names)
    new = term.to_map(type(cmap))
    assert (new.dom, new.cod, new.edge, new.node) == (
        cmap.dom,
        cmap.cod,
        cmap.edge,
        cmap.node,
    )
    return term


def test_term_to_map_identity():
    from discopy.closed import Ty, Variable, Abstraction, Coeval
    X = Ty("X")
    x = Variable(X, "x")
    identity = Abstraction(x, x)
    cmap = identity.to_map()
    assert_freevars_as_domain(identity, cmap)
    assert_trivalent_map(cmap, Ty(), identity.cod, vertices=1)
    assert len(cmap.ports) == 4
    assert isinstance(cmap.boxes[0], Coeval)
    assert cmap.boxes[0].dom == X
    assert cmap.boxes[0].cod == identity.cod @ X
    assert not assert_round_trips_through_term(cmap).freevars


def test_term_to_map_b_combinator():
    from discopy.closed import Ty, Variable, Abstraction, Application, Eval, Coeval

    X, Y, Z = map(Ty, "XYZ")
    x = Variable(Y >> Z, "x")
    y = Variable(X >> Y, "y")
    z = Variable(X, "z")
    b = Abstraction(
        x, Abstraction(y, Abstraction(z, Application(x, Application(y, z))))
    )
    cmap = b.to_map()
    assert_freevars_as_domain(b, cmap)
    assert_trivalent_map(cmap, Ty(), b.cod, vertices=5)
    assert len(cmap.ports) == 16
    assert [type(box) for box in cmap.boxes] == [Eval, Eval, Coeval, Coeval, Coeval]
    assert [len(box.dom) for box in cmap.boxes] == [2, 2, 1, 1, 1]
    assert [len(box.cod) for box in cmap.boxes] == [1, 1, 2, 2, 2]
    assert not assert_round_trips_through_term(cmap).freevars


def test_term_to_map_open_terms_use_domain_boundary():
    from discopy.closed import Ty, Variable, Application, Eval, CombinatorialMap as M

    X, Y = map(Ty, "XY")
    x = Variable(X, "x")
    f = Variable(X >> Y, "f")
    variable = x.to_map()
    assert_freevars_as_domain(x, variable)
    assert variable.dom == X
    assert variable.cod == X
    assert variable == M.id(X)

    application = Application(f, x)
    cmap = application.to_map()
    assert_freevars_as_domain(application, cmap)
    assert_trivalent_map(cmap, (X >> Y) @ X, Y, vertices=1)
    assert isinstance(cmap.boxes[0], Eval)
    assert [port.kind for port in cmap.ports[:2]] == ["input", "input"]
    assert str(assert_round_trips_through_term(cmap, ["f", "x"])) == "f(x)"


def test_whiteboard_term():
    from discopy.closed import Ty

    X, Y, Z = map(Ty, "XYZ")
    term = (Y >> X)(lambda x: Y(lambda y: X(lambda z: z)(x(y))))
    assert term.cod == (Y >> X) >> (Y >> X)
    assert term == term.to_map().to_term()


def test_petersen_shaped_term():
    r"""
    -- typechecks: https://play.haskell.org/saved/7Yl6teux
    petersen :: (((t1 -> t0) -> t5) -> t6)
             -> (t2 -> t3)
             -> (t4 -> t5)
             -> ((t1 -> t0) -> t2)
             -> (t3 -> t4)
             -> t6
    petersen = \ a b c d e -> a (\ f -> c (e (b (d f))))
    """
    from discopy.closed import Ty

    t0, t1, t2, t3, t4, t5, t6 = (Ty(f"x{i}") for i in range(7))

    petersen = (((t1 >> t0) >> t5) >> t6)(
        lambda a: (t2 >> t3)(
            lambda b: (t4 >> t5)(
                lambda c: ((t1 >> t0) >> t2)(
                    lambda d: (t3 >> t4)(
                        lambda e: a((t1 >> t0)(lambda f: c(e(b(d(f))))))
                    )
                )
            )
        )
    )

    assert petersen.cod == (((t1 >> t0) >> t5) >> t6) >> (
        (t2 >> t3) >> ((t4 >> t5) >> (((t1 >> t0) >> t2) >> ((t3 >> t4) >> t6)))
    )

    cmap = petersen.to_map()
    assert len(cmap.ports) == 34
    # cmap.to_hypergraph().simplify().to_diagram().foliation().draw()
    assert_freevars_as_domain(petersen, cmap)
    assert_trivalent_map(cmap, Ty(), petersen.cod, vertices=11)

    roundtrip = cmap.to_term()
    assert petersen == roundtrip
