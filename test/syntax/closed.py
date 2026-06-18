from __future__ import annotations

from pytest import raises

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
    TermBase,
    assert_term_map,
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
    assert str(X("c")) == "c"
    x, y = Variable(X, "x"), Variable(Y, "y")
    assert str(pack(x, y)) == "pack(x, y)"
    assert str(unpack(pack(x, y), (x, y), pack(y, x)))\
        == "unpack(pack(x, y), lambda x, y: pack(y, x))"


def test_term_equality_is_alpha_equivalence():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(X, "y")
    c, d = X("c"), X("d")
    pair = pack(x, y)
    assert X(lambda x: x) == X(lambda y: y)
    assert hash(X(lambda x: x)) == hash(X(lambda y: y))
    assert X(lambda x: (X >> Y)(lambda f: f(x)))\
        == X(lambda y: (X >> Y)(lambda g: g(y)))
    assert x != y
    assert c == X("c")
    assert c != d
    assert Abstraction(x, y) != Abstraction(y, x)
    assert Abstraction(x, x) != Abstraction(Variable(Y, "x"), y)
    assert pack(x) != pair
    assert unpack(pair, (x, y), pack(x, y))\
        != unpack(x, (x, ), x)
    with raises(ValueError):
        unpack(pair, (x, ), x)
    with raises(ValueError):
        unpack(pair, (x, Variable(Y, "z")), pack(y, x))
    assert isinstance(TermBase.alpha_bound(X, 0).name, str)
    assert c.alpha_key(Substitution(()))[0] == "constant"
    assert Application(Variable(X >> X, "f"), x).alpha_key(
        Substitution(()))[0]\
        == "application"
    assert pair.alpha_key(Substitution(()))[0] == "pack"
    assert unpack(pair, (x, y), pair).alpha_key(Substitution(()))[0]\
        == "unpack"


def test_substitution_under_abstraction():
    X = Ty("X")
    x, y, z = (Variable(X, name) for name in "xyz")
    assert Substitution({x: z})(Abstraction(x, x)) == Abstraction(x, x)
    assert Substitution({y: z})(Abstraction(x, y)) == Abstraction(x, z)
    f = Variable(X >> X, "f")
    pair = pack(x, y)
    substitution = Substitution({x: z, y: x})
    assert substitution(Application(f, x)) == Application(f, z)
    assert substitution(pair) == pack(z, x)
    assert substitution(unpack(pair, (x, y), pack(x, y)))\
        == unpack(pack(z, x), (x, y), pack(x, y))
    with raises(ValueError):
        substitution(object())


def test_term_base_helpers_and_errors():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(Y, "y")

    assert Variable.id(X) == Variable(X, "x0")
    assert Variable.id(X @ Y) == pack(Variable(X, "x0"), Variable(Y, "x1"))
    assert Variable.ev(Y, X) == Application(Variable(X >> Y, "f"), x)
    assert x.tensor() is x
    assert x.tensor(y) == pack(x, y)
    assert x.then(Variable(X, "z")) == unpack(x, (Variable(X, "z"), ),
                                             Variable(X, "z"))
    assert Variable.abs(x)(x) == Abstraction(x, x)

    with raises(NotImplementedError):
        X(lambda x, y: x)
    with raises(ValueError):
        X(42)
    with raises(ValueError):
        x.then(y)
    with raises(NotImplementedError):
        x.curry(n=2)
    with raises(ValueError):
        X("constant").curry()


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
    with raises(ValueError):
        unpack(pair, lambda a: a)
    with raises(ValueError):
        unpack(pair)
    with raises(ValueError):
        unpack(x, (x, y), x)


def test_term_failures_and_assertions():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(Y, "y")
    f = Variable(X >> X, "f")
    higher = Variable(X >> (X >> X), "h")

    assert X("c").to_diagram().dom == Ty()
    with raises(ValueError):
        X("c").to_map()
    with raises(TypeError):
        Application(x, x)
    with raises(ValueError):
        Application(f, y)
    with raises(NotImplementedError):
        Application(Application(higher, x), x).to_diagram()
    with raises(ValueError):
        Abstraction(x, y).to_map()
    with raises(ValueError):
        Abstraction(x, Application(Application(higher, x), x)).to_map()

    cmap = CombinatorialMap.id(X)
    with raises(ValueError):
        assert_term_map(cmap, y)
    with raises(ValueError):
        assert_term_map(CombinatorialMap.id(Y), x)
    with raises(ValueError):
        assert_term_map(CombinatorialMap.from_box(Box("f", X @ X, X)), f)


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


def test_term_to_map_identity():
    X = Ty("X")
    x = Variable(X, "x")
    identity = Abstraction(x, x)
    cmap = identity.to_map()
    assert_trivalent_map(cmap, Ty(), identity.cod, vertices=1)
    assert len(cmap.ports) == 4
    assert isinstance(cmap.boxes[0], Coeval)
    assert cmap.boxes[0].dom == X
    assert cmap.boxes[0].cod == identity.cod @ X
    assert cmap.to_term() == X(lambda x: x)


def test_term_to_map_b_combinator():
    X, Y, Z = map(Ty, "XYZ")
    x = Variable(Y >> Z, "x")
    y = Variable(X >> Y, "y")
    z = Variable(X, "z")
    b = Abstraction(
        x, Abstraction(y, Abstraction(z, Application(x, Application(y, z))))
    )
    cmap = b.to_map()
    assert_trivalent_map(cmap, Ty(), b.cod, vertices=5)
    assert len(cmap.ports) == 16
    assert [type(box) for box in cmap.boxes] == [Eval, Eval, Coeval, Coeval, Coeval]
    assert [len(box.dom) for box in cmap.boxes] == [2, 2, 1, 1, 1]
    assert [len(box.cod) for box in cmap.boxes] == [1, 1, 2, 2, 2]
    assert cmap.to_term() == b


def test_term_to_map_open_terms_use_domain_boundary():
    X, Y = map(Ty, "XY")
    x = Variable(X, "x")
    f = Variable(X >> Y, "f")
    variable = x.to_map()
    print(f"{variable.dom!r}, {variable.cod!r}")
    assert variable.dom == X
    assert variable.cod == X
    assert variable == CombinatorialMap.id(X)

    application = Application(f, x)
    cmap = application.to_map()
    assert_trivalent_map(cmap, (X >> Y) @ X, Y, vertices=1)
    assert isinstance(cmap.boxes[0], Eval)
    assert [port.kind for port in cmap.ports[:2]] == ["input", "input"]
    assert cmap.to_term(["f", "x"]) == application


def test_whiteboard_term():
    X, Y, Z = map(Ty, "XYZ")
    term = (Y >> X)(lambda x: Y(lambda y: X(lambda z: z)(x(y))))
    assert term.cod == (Y >> X) >> (Y >> X)
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

    t0, t1, t2, t3, t4, t5, t6 = (Ty(f"t{i}") for i in range(7))

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
    assert_trivalent_map(cmap, Ty(), petersen.cod, vertices=11)

    roundtrip = cmap.to_term()
    assert petersen == roundtrip
