from __future__ import annotations

from discopy.closed import (
    Application,
    Abstraction,
    Variable,
    Ty,
    Functor,
    Box,
    Eval,
    Coeval,
    CombinatorialMap,
    Substitution,
    _alpha_bound,
    pack,
    unpack,
)


def test_exp():
    X, Y = Ty("X"), Ty("Y")
    assert X >> Y == Y**X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_str():
    X, Y = Ty("X"), Ty("Y")
    f = X(lambda x: (X >> Y)(lambda y: y(x)))
    assert str(f) == "X(lambda x: (X >> Y)(lambda y: y(x)))"


def test_term_equality_is_alpha_equivalence():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(X, "y")
    assert X(lambda x: x) == X(lambda y: y)
    assert hash(X(lambda x: x)) == hash(X(lambda y: y))
    assert X(lambda x: (X >> Y)(lambda f: f(x)))\
        == X(lambda y: (X >> Y)(lambda g: g(y)))
    assert x != y
    assert Abstraction(x, y) != Abstraction(y, x)
    assert isinstance(_alpha_bound(X, 0).name, str)


def test_substitution_under_abstraction():
    X = Ty("X")
    x, y, z = (Variable(X, name) for name in "xyz")
    assert Substitution({x: z})(Abstraction(x, x)) == Abstraction(x, x)
    assert Substitution({y: z})(Abstraction(x, y)) == Abstraction(x, z)


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box("f", y, x >> z), Box("g", x @ y, z)

    from discopy.python import Function

    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={
            f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j),
        },
        cod=Function,
    )

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


def test_pack_unpack_terms():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(Y, "y")

    pair = pack(x, y)
    assert pair.dom == X @ Y
    assert pair.cod == X @ Y
    assert pair.to_diagram().dom == X @ Y
    assert pair.to_diagram().cod == X @ Y
    assert pair.to_map() == CombinatorialMap.id(X @ Y)

    swap = unpack(pair, lambda a, b: pack(b, a))
    assert swap.dom == X @ Y
    assert swap.cod == Y @ X
    assert swap.to_diagram().dom == X @ Y
    assert swap.to_diagram().cod == Y @ X
    assert swap.to_map().dom == X @ Y
    assert swap.to_map().cod == Y @ X

    assert unpack(pair, lambda a, b: pack(a, b))\
        == unpack(pair, lambda c, d: pack(c, d))


def test_python_Func():
    x, y, z = map(Ty, "xyz")
    f, g = Box("f", y, x >> z), Box("g", x @ y, z)

    from discopy.python import Function

    ob = lambda typ: {"x": complex, "y": bool, "z": float}[str(typ)]

    F = Functor(
        ob=ob,
        ar={
            f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j),
        },
        cod=Function,
    )

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


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
    X, Y = map(Ty, "XY")
    x = Variable(X, "x")
    f = Variable(X >> Y, "f")
    variable = x.to_map()
    assert_freevars_as_domain(x, variable)
    assert variable.dom == X
    assert variable.cod == X
    assert variable == CombinatorialMap.id(X)

    application = Application(f, x)
    cmap = application.to_map()
    assert_freevars_as_domain(application, cmap)
    assert_trivalent_map(cmap, (X >> Y) @ X, Y, vertices=1)
    assert isinstance(cmap.boxes[0], Eval)
    assert [port.kind for port in cmap.ports[:2]] == ["input", "input"]
    assert str(assert_round_trips_through_term(cmap, ["f", "x"])) == "f(x)"


def test_whiteboard_term():
    X, Y, Z = map(Ty, "XYZ")
    term = (Y >> X)(lambda x: Y(lambda y: X(lambda z: z)(x(y))))
    assert term.cod == (Y >> X) >> (Y >> X)
    term.to_map().draw()
    assert term == term.to_map().to_term()


def test_petersen_term():
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
    cmap.draw()
    assert len(cmap.ports) == 34
    # cmap.draw()
    # cmap.to_hypergraph().simplify().to_diagram().foliation().draw()
    assert_freevars_as_domain(petersen, cmap)
    assert_trivalent_map(cmap, Ty(), petersen.cod, vertices=11)

    roundtrip = cmap.to_term()
    assert petersen == roundtrip
