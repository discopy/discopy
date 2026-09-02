"""B72: a whiskered swap is a Permutation that functors cannot map, three ways (discopy/symmetric.py:653).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import balanced
from discopy.matrix import Matrix
from discopy.monoidal import Equation
from discopy.symmetric import PRO, Diagram, Functor, Swap, Ty

x, y, z = map(Ty, "xyz")


def test_b72_functor_into_matrix_on_bare_swap_control_passes():
    functor = Functor({x: 2, y: 3, z: 4}, {}, cod=Matrix)
    assert functor(Swap(x, y)) == Matrix.swap(2, 3)


def test_b72_functor_into_matrix_on_whiskered_swap():
    functor = Functor({x: 2, y: 3, z: 4}, {}, cod=Matrix)
    assert functor(Swap(x, y) @ z) == Matrix.swap(2, 3) @ Matrix.id(4)


def test_b72_whiskered_swap_to_braided():
    diagram = (Swap(x, y) @ z).to_braided()
    assert [type(box) for box in diagram.boxes] == [balanced.DualRailBraid]


def test_b72_pro_ob_map_widening_a_wire():
    functor = Functor({PRO(1): x @ y}, {}, cod=Diagram)
    swap = Diagram.swap(PRO(1), PRO(1)) @ PRO(1)
    assert Equation(functor(swap), Diagram.swap(x @ y, x @ y) @ (x @ y))
