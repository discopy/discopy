from __future__ import annotations

import pytest

from discopy.closed import Application, Abstraction, Variable, Exp, Ty, Functor


def test_exp():
    X, Y = Ty("X"), Ty("Y")
    assert X >> Y == Y**X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_str():
    X, Y = Ty("X"), Ty("Y")
    f = X(lambda x: (X >> Y)(lambda y: y(x)))
    assert str(f) == "X(lambda x: (X >> Y)(lambda y: y(x)))"


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box("f", y, x >> z), Box("g", x @ y, z)

    from discopy.python import Function
    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j)},
        cod=Function)

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


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


def test_term_to_map_b_combinator():
    X, Y, Z = map(Ty, "XYZ")
    x = Variable(Y >> Z, "x")
    y = Variable(X >> Y, "y")
    z = Variable(X, "z")
    b = Abstraction(
        x, Abstraction(y, Abstraction(z, Application(x, Application(y, z)))))
    cmap = b.to_map()
    assert_freevars_as_domain(b, cmap)
    assert_trivalent_map(cmap, Ty(), b.cod, vertices=5)
    assert len(cmap.ports) == 16
    assert [type(box) for box in cmap.boxes] == [
        Eval, Eval, Coeval, Coeval, Coeval]
    assert [len(box.dom) for box in cmap.boxes] == [2, 2, 1, 1, 1]
    assert [len(box.cod) for box in cmap.boxes] == [1, 1, 2, 2, 2]


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


def test_term_to_map_rejects_non_linear_terms():
    X, Y, Z = map(Ty, "XYZ")
    x, y = Variable(X, "x"), Variable(Y, "y")
    with pytest.raises(ValueError):
        Abstraction(x, Abstraction(y, x)).to_map()

    f = Variable(X >> Y, "f")
    h = Variable((X >> Y) >> ((X >> Y) >> Z), "h")
    duplicate = Abstraction(
        h, Abstraction(f, Application(Application(h, f), f)))
    with pytest.raises(ValueError):
        duplicate.to_map()


def test_term_to_map_rejects_constants():
    X = Ty("X")
    with pytest.raises(ValueError):
        X("c").to_map()


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
    x0, x1, x2, x3, x4, x5, x6 = map(Ty, "0123456")
    a, b, c, d, e, f = (
        Variable(ty, name)
        for name, ty in (
            ("a", Exp(x6, Exp(x5, Exp(x0, x1)))),
            ("b", Exp(x3, x2)),
            ("c", Exp(x5, x4)),
            ("d", Exp(x2, Exp(x0, x1))),
            ("e", Exp(x4, x3)),
            ("f", Exp(x0, x1)),
        )
    )

    petersen = Abstraction(
        a,
        Abstraction(
            b,
            Abstraction(
                c,
                Abstraction(
                    d,
                    Abstraction(
                        e,
                        a(Abstraction(f, c(e(b(d(f)))))))))))

    assert petersen.cod == Exp(
        Exp(
            Exp(
                Exp(
                    Exp(
                        x6,
                        Exp(x4, x3)),
                    Exp(x2, Exp(x0, x1))),
                Exp(x5, x4)),
            Exp(x3, x2)),
        Exp(x6, Exp(x5, Exp(x0, x1))))

    cmap = petersen.to_map()
    # cmap.to_hypergraph().simplify().to_diagram().foliation().draw()
    print(cmap.to_hypergraph().simplify().to_diagram())
    assert_freevars_as_domain(petersen, cmap)
    assert_trivalent_map(cmap, Ty(), petersen.cod, vertices=11)
    assert len(cmap.ports) == 34
