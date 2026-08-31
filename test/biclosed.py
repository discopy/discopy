from discopy.biclosed import *
from discopy import cat
from pytest import raises


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert isinstance(x.over_factory(x, y), cat.Ob)
    assert not isinstance(x.over_factory(x, y), Ty)
    assert x.over(y) == x << y
    assert isinstance(x ** y, Ty)
    assert not isinstance((x ** y).inside[0], Ty)
    assert isinstance(x << y, Ty)
    assert not isinstance((x << y).inside[0], Ty)
    assert (x << y).inside == (Over(x, y), )
    assert repr(Over(x, y))\
        == "biclosed.Over(biclosed.Ty(cat.Ob('x')), biclosed.Ty(cat.Ob('y')))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
    assert isinstance(y.under_factory(y, x), cat.Ob)
    assert not isinstance(y.under_factory(y, x), Ty)
    assert y.under(x) == x >> y
    assert isinstance(x >> y, Ty)
    assert not isinstance((x >> y).inside[0], Ty)
    assert (x >> y).inside == (Under(y, x), )
    assert repr(Under(x, y))\
        == "biclosed.Under(biclosed.Ty(cat.Ob('x')), biclosed.Ty(cat.Ob('y')))"
    assert {Under(x, y): 42}[Under(x, y)] == 42
    assert Under(x, y) != Over(x, y)


def test_Term():
    x, y = Ty('x'), Ty('y')
    assert isinstance((x << y).inside[0], cat.Ob)
    assert not isinstance((x << y).inside[0], Ty)
    f, g = (x << y)("f"), (y >> x)("g")
    a = y("a")

    assert isinstance(f, TermBase)
    assert f(a).cod == x
    assert f(a).eval() == f @ a >> Eval(x << y)
    assert a(g, left=True).cod == x
    assert a(g, left=True).eval() == a @ g >> Eval(y >> x)

    var = Variable('var', y)
    assert Abstraction(var, f(var)).cod == x << y
    assert Abstraction(var, var(g, left=True), left=True).cod == y >> x


def test_Term_str():
    X, Y = Ty('X'), Ty('Y')
    f, g = (Y << X)("f"), (X >> Y)("g")
    x, y = X("x"), Variable("y", X)
    assert str(f(x)) == "(Y << X)('f')(X('x'))"
    assert str(x(g, left=True)) == "X('x')((X >> Y)('g'), left=True)"
    assert str(X(lambda y: f(y))) == "X(lambda y: (Y << X)('f')(y))"
    assert str(f(y)) == "(Y << X)('f')(y)"


def test_Term_linear_planar():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = (x << y)("f"), (y >> x)("g")
    fvar = Variable('fvar', x << y)
    gvar = Variable('gvar', y >> x)
    h = ((x << y) << y)("h")
    var = Variable('var', y)

    with raises(ValueError):
        h(var)(var)
    with raises(ValueError):
        z(lambda u, left=True: f(var))

    eta_over = Abstraction(var, fvar(var))
    eta_under = Abstraction(var, var(gvar, left=True), left=True)
    assert (eta_over.dom, eta_over.cod) == (x << y, x << y)
    assert (eta_under.dom, eta_under.cod) == (y >> x, y >> x)


def test_Application_freevars_order():
    """The free variables of a term are ordered like the wires of its dom."""
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    a, b = Variable('a', y), Variable('b', x)
    func = a((y >> (z << x))("h"), left=True)

    under = b((x >> (y >> z))("k"), left=True)
    for term in [func(b), a(under, left=True)]:
        assert term.dom == Ty().tensor(*[var.cod for var in term.freevars])
        assert term.freevars == [a, b]


def test_Abstraction_well_typed():
    """Abstraction agrees with eval on both dom and cod, nesting included."""
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    a, b = Variable('a', y), Variable('b', x)
    term = a((y >> (z << x))("h"), left=True)(b)

    inner = Abstraction(b, term)
    assert (inner.dom, inner.cod) == (y, z << x)
    outer = Abstraction(a, inner, left=True)
    assert (outer.dom, outer.cod) == (Ty(), y >> (z << x))
    for abstraction in [inner, outer]:
        evaluated = abstraction.eval()
        assert (evaluated.dom, evaluated.cod) == (abstraction.dom,
                                                  abstraction.cod)

    with raises(ValueError):
        Abstraction(a, term)


def test_to_term_round_trip():
    X, Y, N, S = Ty('X'), Ty('Y'), Ty('N'), Ty('S')
    f, g, h = (X >> Y)("f"), (Y << X)("g"), ((X >> Y) << X)("h")
    x = Variable("x", X)
    Alice, loves, Bob = N("Alice"), ((N >> S) << N)("loves"), N("Bob")
    for term in [
            X("c"),
            x(f, left=True),
            g(x),
            X(lambda z, left=True: z(f, left=True)),
            X(lambda z: g(z)),
            Alice(loves(Bob), left=True),
            X(lambda a: h(a)),
            Abstraction(x, Variable("fv", Y << X)(x)),
            Abstraction(x, x(Variable("gv", X >> Y), left=True), left=True)]:
        result = term.eval().to_term()
        assert result == term and str(result) == str(term)


def test_to_term_varnames():
    X, Y = Ty('X'), Ty('Y')
    f = (X >> Y)("f")
    diagram = X(lambda my_var, left=True: my_var(f, left=True)).eval()
    assert [
        getattr(obj, "varname", None)
        for obj in diagram.boxes[0].arg.dom.inside] == ["my_var"]


def test_to_term_fresh_names():
    X, Y = Ty('X'), Ty('Y')
    diagram = Curry(Eval(Y << X))  # hand-built, hence no varname attributes
    term = diagram.to_term()
    assert isinstance(term, Abstraction)
    assert term.eval() == diagram


def test_to_term_multi_object_variable():
    x, y = Ty('x'), Ty('y')
    pair = Variable("pair", x @ y)
    assert pair.eval().to_term() == pair


def test_to_term_errors():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    with raises(ValueError):
        Id(x @ y).to_term()  # two variables are not a single term
    with raises(ValueError):
        Box('f', x, y).to_term()  # a box with inputs is not a constant
    with raises(ValueError):
        Coeval(y << x).to_term()
    with raises(ValueError):
        Variable('v', x).to_term()  # a term box with inputs
    with raises(ValueError):  # a box cutting through the wires of a term
        ((x @ y)("c") >> Box('g', x, x) @ y).to_term()
    with raises(ValueError):  # three terms fed to a single evaluation
        (x("a") @ y("b") @ ((x @ y) >> z)("c")
         >> Eval((x @ y) >> z)).to_term()
    with raises(ValueError):  # two terms that do not split as func and args
        (x("a") @ (y @ ((x @ y) >> z))("pair")
         >> Eval((x @ y) >> z)).to_term()
    with raises(ValueError):  # foliated layers are not supported
        (Box('a', Ty(), x) @ Box('b', Ty(), y)).foliation().to_term()



def test_to_rigid():
    from discopy import rigid

    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> Diagram.ev(x, y, left=True)
    assert Diagram.to_rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert Diagram.to_rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)


def test_to_compact():
    x, y, z = map(Ty, "xyz")
    f = Box("f", x @ y, z)
    for left in (True, False):
        source = f.curry(left=left)
        assert source.to_compact() == (
            f.to_map() >> CMap.ev(z, y if left else x, left).dagger()).trace(
                left=not left)
        assert source.to_map().to_compact() == source.to_compact()
        assert not any(isinstance(box, Curry)
                       for box in source.to_compact().boxes)


def test_fresh_names_avoid_annotated_names():
    from discopy.biclosed import _dom_to_variables
    clash = f"x{int(fresh_name()[1:]) + 1}"
    diagram = Diagram.id(annotate(Ty("x"), clash) @ Ty("y"))
    variables = _dom_to_variables(diagram)
    assert len({variable.name for variable in variables}) == 2
