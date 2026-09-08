from pytest import raises

from discopy.interaction import Diagram, Id, Ty


def test_Ty_repr():
    t = Ty[int](positive=1, negative=2)
    assert repr(t)\
        == str(t) == "interaction.Ty[int](positive=1, negative=2)"


def test_Ty_str():
    x, y, z, w = map(Ty, "xyzw")
    assert str(x @ -y @ z @ -w) == "x @ z @ -y @ -w"


def test_Ty_unit():
    assert Ty.unit() == Ty()
    assert Ty[int].unit() == Ty[int]() == Ty[int](0, 0)
    assert Ty[tuple].unit() == Ty[tuple]((), ())


def test_Ty_negatives():
    assert Ty.negatives is reversed
    x, y = Ty[tuple]((1, ), (2, )), Ty[tuple]((3, ), (4, ))
    assert x @ y == Ty[tuple]((1, 3), (4, 2))
    with raises(TypeError):
        x @ Ty[int](1, 2)


def test_Diagram_permutation():
    from discopy import compact
    x0, x1, y0, y1, z0, z1 = map(
        compact.Ty, ("x0", "x1", "y0", "y1", "z0", "z1"))
    x, y, z = (
        Ty[compact.Ty](x0, x1),
        Ty[compact.Ty](y0, y1),
        Ty[compact.Ty](z0, z1))
    diagram = Diagram[compact.Diagram]
    permutation = diagram.permutation([2, 0, 1], [x, y, z])
    assert permutation.dom == x @ y @ z
    assert permutation.cod == z @ x @ y
    assert diagram.permutation([0, 1, 2], [x, y, z])\
        == diagram.id(x @ y @ z)
    with raises(ValueError):
        diagram.permutation([1, 0], [x, y, z])


def test_snake_equations():
    from discopy import symmetric
    x = Ty[symmetric.Ty](symmetric.Ty('a'), symmetric.Ty('b'))
    D = Diagram[symmetric.Diagram]
    hypergraph = lambda diagram: diagram.inside.to_hypergraph()
    left_snake = D.caps(x, -x) @ D.id(x) >> D.id(x) @ D.cups(-x, x)
    right_snake = D.id(x) @ D.caps(-x, x) >> D.cups(x, -x) @ D.id(x)
    assert hypergraph(left_snake) == hypergraph(D.id(x))
    assert hypergraph(right_snake) == hypergraph(D.id(x))


def test_ValueError():
    from discopy.ribbon import Ty as T, Diagram as D, Box as B
    x, y, z = map(Ty[T], "xyz")
    f = B('f', T('x'), T('y'))
    with raises(ValueError):
        Diagram[D](f, x, z)
    with raises(ValueError):
        Diagram[D](f, z, y)


def test_IndexError():
    with raises(IndexError):
        return Id()[:]
