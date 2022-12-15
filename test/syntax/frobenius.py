from pytest import raises

from discopy.frobenius import *


def test_Functor_call():
    x = Ty('x')
    cup, cap = Cup(x, x.r), Cap(x.r, x)
    box1, box2 = Box('box', x, x), Box('box', x @ x, x @ x)
    spider = Spider(0, 2, x)
    F = Functor(lambda x: x @ x, {box1: box2})
    assert F(cup) == Id(x) @ cup @ Id(x.r) >> cup
    assert F(cap) == Id(x.r) @ cap @ Id(x) << cap
    assert F(box1) == box2
    assert F(box1.l) == box2.l and F(box1.r) == box2.r
    assert F(spider) == spider @ spider >> Id(x) @ Swap(x, x) @ Id(x)
    with raises(TypeError):
        F(F)


def test_spider_adjoint():
    n = Ty('n')
    one = Box('one', Ty(), n)
    two = Box('two', Ty(), n)
    diagram = one @ two >> Spider(2, 1, n)

    assert diagram.r == diagram.l == Spider(1, 2, n) >> two.r @ n >> one.r


def test_spider_factory():
    a, b, c = map(Ty, 'abc')
    ts = [a, a @ b, a @ b @ c]
    for i in range(5):
        for j in range(5):
            for t in ts:
                s = Diagram.spiders(i, j, t)
                for k, ob in enumerate(t):
                    assert all(map(ob.__eq__, s.dom[k::len(t)]))
                    assert all(map(ob.__eq__, s.cod[k::len(t)]))


def test_spider_decomposition():
    n = Ty('n')

    assert Spider(0, 0, n).unfuse() == Spider(0, 1, n) >> Spider(1, 0, n)
    assert Spider(1, 0, n).unfuse() == Spider(1, 0, n)
    assert Spider(1, 1, n).unfuse() == Id(n)
    assert Spider(2, 1, n).unfuse() == Spider(2, 1, n)

    # 5 is the smallest number including both an even and odd decomposition
    assert Spider(5, 1, n).unfuse() == (Spider(2, 1, n) @ Spider(2, 1, n)
                                           @ Id(n) >> Spider(2, 1, n) @ Id(n)
                                           >> Spider(2, 1, n))
