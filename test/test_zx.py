from pytest import raises
from discopy import *
from discopy.quantum.zx import *


def test_Diagram():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    repr(bialgebra) == "zx.Diagram(dom=PRO(2), cod=PRO(2), "\
                       "boxes=[Z(1, 2), Z(1, 2), SWAP, X(2, 1), X(2, 1)], "\
                       "offsets=[0, 2, 1, 0, 1])"
    str(bialgebra) == "Z(1, 2) @ Id(1) >> Id(2) @ Z(1, 2)"\
                      ">> Id(1) @ SWAP @ Id(1)"\
                      ">> X(2, 1) @ Id(2) >> Id(1) @ X(2, 1)"

def test_Swap():
    x = Ty('x')
    with raises(TypeError):
        Swap(x, x)
    with raises(TypeError):
        Box('f', PRO(0), x)


def test_Spider():
    assert repr(Z(1, 2, 3)) == "Z(1, 2, 3)"
    assert repr(Y(1, 2, 3)) == "Y(1, 2, 3)"
    assert repr(X(1, 2, 3)) == "X(1, 2, 3)"
    for spider in [Z, Y, X]:
        assert spider(1, 2, 3).phase == 3
        assert spider(1, 2, 3j).dagger() == spider(2, 1, -3j)


def test_H():
    assert repr(H) == str(H) == "H"
    assert H[::-1] == H


def test_Sum():
    assert Z(1, 1) + Z(1, 1) >> Z(1, 1) == sum(2 * [Z(1, 1) >> Z(1, 1)])


def test_scalar():
    assert scalar(1j)[::-1] == scalar(-1j)


def test_Functor():
    x = Ty('x')
    f = rigid.Box('f', x, x)
    F = rigid.Functor(
        ob=lambda _: PRO(1),
        ar=lambda f: Z(len(f.dom), len(f.cod)),
        ob_factory=PRO,
        ar_factory=Diagram)
    assert F(f) == Z(1, 1)
    assert F(rigid.Swap(x, x)) == Diagram.permutation([1, 0]) == SWAP
    assert F(Cup(x.l, x)) == Z(2, 0)
    assert F(Cap(x.r, x)) == Z(0, 2)
    assert F(f + f) == Z(1, 1) + Z(1, 1)


def test_subs():
    from sympy.abc import phi, psi
    assert Z(3, 2, phi).subs(phi, 1) == Z(3, 2, 1)
    assert scalar(phi).subs(phi, psi) == scalar(psi)


def test_grad():
    from sympy.abc import phi, psi
    assert not scalar(phi).grad(psi) and scalar(phi).grad(phi) == scalar(1)
    assert not Z(1, 1, phi).grad(psi)
    assert Z(1, 1, phi).grad(phi) == scalar(0.5j) @ Z(1, 1, phi - .5)
    assert (Z(1, 1, phi / 2) >> Z(1, 1, phi + 1)).grad(phi)\
        == (scalar(0.25j) @ Z(1, 1, phi / 2 - .5) >> Z(1, 1, phi + 1))\
           + (Z(1, 1, phi / 2) >> scalar(0.5j) @ Id(1) >> Z(1, 1, phi + .5))


def test_to_pyzx_errors():
    with raises(TypeError):
        Diagram.to_pyzx(quantum.H)


def test_to_pyzx():
    Diagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2) >> SWAP


def test_from_pyzx_errors():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    graph = bialgebra.to_pyzx()
    graph.inputs = graph.outputs = []
    with raises(ValueError):  # missing_boundary
        Diagram.from_pyzx(graph)
    graph.auto_detect_io()
    with raises(ValueError):  # duplicate_boundary
        Diagram.from_pyzx(graph)



def test_backnforth_pyzx():
    from pyzx import Graph
    path = 'test/src/zx-graph.json'
    graph = Graph.from_json(open(path).read())
    diagram = Diagram.from_pyzx(graph)
    backnforth = lambda diagram: Diagram.from_pyzx(diagram.to_pyzx())
    assert backnforth(diagram) == diagram


def test_circui2zx():
    circuit = Ket(0, 0) >> quantum.H @ Rx(0) >> CRz(0) >> CRx(0) >> CU1(0)
    circuit2zx(circuit) == Diagram(
        dom=PRO(0), cod=PRO(2), boxes=[
            X(0, 1), X(0, 1), H, X(1, 1),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0),
            X(1, 2), X(1, 2), Z(2, 1), X(1, 0),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0)],
        offsets=[0, 1, 0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1])
