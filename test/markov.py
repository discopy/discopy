from __future__ import annotations
from pytest import raises

from discopy.markov import *
from discopy import *
from discopy.utils import AxiomError, dumps, loads


def test_spider_factory():
    with raises(ValueError):
        Diagram.spider_factory(2, 2, Ty('x'))
    with raises(ValueError):
        Diagram.spider_factory(2, 1, Ty('x'))


def test_Copy_dagger():
    with raises(AxiomError):
        Copy(Ty('x')).dagger()


def test_Discard():
    assert isinstance(Discard(Ty('x')), Discard)
    assert isinstance(Copy(Ty('x'), n=0), Discard)


def test_equations():
    x = Ty('x')
    copy, discard = Copy(x), Copy(x, 0)
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)

    add >> copy, copy @ copy >> x @ Swap(x, x) @ x >> add @ add
    add >> discard, discard @ discard
    zero >> discard, Diagram.id(Ty())
    copy >> minus @ x >> add, discard >> zero, copy >> x @ minus >> add

    Diagram.id(x)
    x @ zero >> x @ copy >> add @ x >> discard @ x
    x @ zero @ zero >> discard @ discard @ x
    discard >> zero


def test_neural_network():
    x = Ty('x')
    add = lambda n: Box('$+$', x ** n, x)
    ReLU = Box('$\\sigma$', x, x)
    weights = [Box('w{}'.format(i), x, x) for i in range(4)]
    bias = Box('b', Ty(), x)

    network = Diagram.copy(x @ x, 2)\
    >> Diagram.tensor(*weights) @ bias >> add(5) >> ReLU

    F = Functor(ob_map={x: int}, ar_map={
            add(5): lambda *xs: sum(xs),
            ReLU: lambda x: max(0, x),
            bias: lambda: -1, **{
                weight: lambda x, w=w: x * w
                for weight, w in zip(weights, range(4))}},
        cod=python.Function)

    assert F(network)(42, 43) == max(0, sum([42 * 0, 43 * 1, 42 * 2, 43 * 3, -1]))


def test_Permutation():
    x, y, z = map(Ty, "xyz")
    assert Diagram.permutation_factory is Permutation
    perm = Permutation(x @ y @ z, [2, 0, 1])
    assert isinstance(perm, Box) and perm.cod == z @ x @ y
    assert perm.fun == perm.perm
    assert Equation(perm >> perm.dagger(), Id(x @ y @ z))
    assert isinstance(perm.inside[0], Layer)
    assert all(isinstance(f, Function) for f in Box('f', x, y).inside[0][::2])
    assert Permutation(x @ y, [1, 0]) != Swap(x, y)
    assert Equation(perm, perm.to_swaps())


def test_Function():
    x, y = Ty('x'), Ty('y')
    fun = Function(x @ y, [1, 0, 0])
    assert fun.cod == y @ x @ x
    assert Equation(fun, Swap(x, y) >> y @ Copy(x))
    assert Equation(fun.to_copies(), fun)
    assert Function(x @ y, [0, 1]).is_identity
    assert loads(dumps(fun)) == fun
    assert Function(x @ y, iter([1, 0, 0])) == fun
    with raises(ValueError):
        Function(x, [1])


def test_Function_dagger():
    x, y = Ty('x'), Ty('y')
    assert Function(x @ y, [1, 0]).dagger() == Function(y @ x, [1, 0])
    with raises(AxiomError):
        Function(x, [0, 0]).dagger()


def test_Function_tensor():
    x, y = Ty('x'), Ty('y')
    copy, swap = Function(x, [0, 0]), Function(x @ y, [1, 0])
    assert copy @ swap == Function(x @ x @ y, [0, 0, 2, 1])
    assert copy @ y == Function(x @ y, [0, 0, 1])
    assert y @ copy == Function(y @ x, [0, 1, 1])
    from discopy import symmetric
    perm = symmetric.Permutation(y, [0])
    assert isinstance(copy @ perm, Function)
    assert Permutation(y, [0]) @ copy == Function(y @ x, [0, 1, 1])


def test_Layer():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    layer = Layer(x, f, y)
    assert layer.functions == layer.permutations
    assert all(isinstance(g, Function) for g in layer[::2])
    fun = Function(x @ y, [1, 0, 0])
    assert Layer(fun).is_permutation and Layer(fun).permutation == fun
    with raises(AxiomError):
        Layer(fun).dagger()


def test_from_function():
    x, y = Ty('x'), Ty('y')
    assert Diagram.from_function([1, 0, 1], x @ y) == Function(x @ y, [1, 0, 1])
    assert Diagram.from_function([0, 1], x @ y) == Id(x @ y)
    assert Diagram.from_function([0, 0]) == Function(PRO(1), [0, 0])


def test_discard():
    x, y = Ty('x'), Ty('y')
    assert Equation(Diagram.discard(x @ y), Copy(x, 0) @ Copy(y, 0))


def test_function():
    x, y = Ty('x'), Ty('y')
    assert Diagram.function([0, 1], x @ y) == Id(x @ y)
    assert Equation(Diagram.function([0, 0], x), Copy(x))
    with raises(ValueError):
        Diagram.function([2], x)


def test_from_function_fallback():
    from discopy import closed
    x = closed.Ty('x')
    diagram = closed.Diagram.from_function([0, 0], x)
    assert isinstance(diagram, closed.Diagram)
    assert not isinstance(diagram, Function)


def test_Function_pickle():
    import pickle
    x, y = Ty('x'), Ty('y')
    for fun in (Function(x @ y, [0, 1]), Function(x @ y, [1, 0, 0])):
        assert pickle.loads(pickle.dumps(fun)) == fun


def test_Function_size():
    assert Function(Ty('x'), [0, 0]).size == 0


def test_Permutation_tensor_Function():
    x, y = Ty('x'), Ty('y')
    result = Permutation(x @ y, [1, 0]).tensor(Function(x, [0, 0]))
    assert result == Function(x @ y @ x, [1, 0, 2, 2])
    assert Permutation(x @ y, [1, 0]) @ y == Permutation(x @ y @ y, [1, 0, 2])


def test_Function_functor():
    x, y = Ty('x'), Ty('y')
    fun = Function(x @ y, [1, 0, 0])
    F = Functor({x: int, y: str}, {}, cod=python.Function)
    assert F(fun)(42, 'a') == ('a', 42, 42)
    G = Functor(lambda ob: ob, lambda box: box)
    assert Equation(G(fun), fun)
