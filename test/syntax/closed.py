from discopy.closed import *


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert repr(Over(x, y))\
        == "closed.Over(closed.Ty(cat.Ob('x')), closed.Ty(cat.Ob('y')))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
    assert repr(Under(x, y))\
        == "closed.Under(closed.Ty(cat.Ob('x')), closed.Ty(cat.Ob('y')))"
    assert {Under(x, y): 42}[Under(x, y)] == 42
    assert Under(x, y) != Over(x, y)


def test_to_rigid():
    from discopy import rigid

    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> Diagram.ev(x, y)
    assert Diagram.to_rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert Diagram.to_rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', y, z << x), Box('g', y, z >> x)

    from discopy.python import Function
    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda y: lambda z: z + 1j if y else -1j},
        cod=Category(tuple[type, ...], Function))

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.uncurry(left=False).curry(left=False))(True)(1.2) == F(g)(True)(1.2)


def test_constant_eval():
    x, y = map(Ty, "xy")
    f = Box("f", x, y)
    g = Box("g", x, Ty() >> y)
    h = Box("h", Ty(), x >> y)

    from discopy.python import Function
    F = Functor(
        ob={x: str, y: int},
        ar={f: lambda a: len(a),
            g: lambda a: lambda: len(a),
            h: lambda: lambda a: len(a)},
        cod=Category(tuple[type, ...], Function))
    

    g_eval = g >> Eval(Ty() >> y)
    h_eval = x @ h >> Eval(x >> y)

    test_str = "Slicing two ways!"
    assert F(f)(test_str) == len(test_str)
    assert F(f)(test_str) == F(g_eval)(test_str)
    assert F(g_eval)(test_str) == F(h_eval)(test_str)


def test_partial_eval():
    x, y, z = map(Ty, "xyz")
    
    f = Box("f", x @ y, z)
    g = Box("g", Ty(), (x @ y) >> z)
    g2 = Box("g2", Ty(), y >> (x >> z))
    h = Box("h", y, x >> z)

    g_eval = x @ y @ g >> Eval(x @ y >> z)
    h_eval = x @ h >> Eval(x >> z)
    g_partial_eval = (x @ ((y @ g2) >> Eval((y) >> (x >> z))) ) >> Eval(x >> z)

    from discopy.python import Function
    F = Functor(
        ob={x: str, y: int, z: str},
        ar={f: lambda a, b: a[:b],
            g: lambda: lambda a, b: a[:b],
            g2: lambda: lambda b: lambda a: a[:b],
            h: lambda b: lambda a: a[:b],
            },
        cod=Category(tuple[type, ...], Function))
    
    test_str = "Partial evaluator"
    str_slice = 7
    
    assert F(f)(test_str, str_slice) == "Partial"
    assert F(f)(test_str, str_slice) == F(g_eval)(test_str, str_slice)
    assert F(g_eval)(test_str, str_slice) == F(g_partial_eval)(test_str, str_slice)
    assert F(g_partial_eval)(test_str, str_slice) == F(h_eval)(test_str, str_slice)

