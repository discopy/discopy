"""B51: balanced.Functor sends Twist(x).dagger() to Twist(x), DualRail too (discopy/balanced.py:319).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy import balanced, ribbon
from discopy.hopf import Algebra, Double, Functor, Intertwiner, Representation

x = balanced.Ty('x')


def test_b51_identity_functor_keeps_twist_dagger():
    twist = balanced.Twist(x).dagger()
    assert balanced.Functor.id(balanced.Diagram)(twist) == twist


def test_b51_identity_functor_keeps_twist_control_passes():
    twist = balanced.Twist(x)
    assert balanced.Functor.id(balanced.Diagram)(twist) == twist


def test_b51_dual_rail_keeps_twist_dagger():
    plain, daggered = balanced.Twist(x), balanced.Twist(x).dagger()
    assert daggered.to_braided().boxes[0].is_dagger
    assert daggered.to_braided() != plain.to_braided()


def test_b51_hopf_functor_twist_then_dagger_is_identity():
    D = Double(Algebra.taft(3))
    y = ribbon.Ty('y')
    F = Functor(ob_map={y: Representation[D].regular()}, ar_map={},
                cod=Intertwiner[D])
    twist = ribbon.Twist(y)
    array = F(twist >> twist.dagger()).eval(dtype=complex).array
    assert np.allclose(array.reshape(81, 81), np.eye(81))
