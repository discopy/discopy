"""B69: daggered braids, delayed boxes and head/tail objects do not survive loads(dumps) (discopy/utils.py:432, feedback.py:219).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import balanced, braided, feedback, ribbon
from discopy.utils import dumps, loads

x = feedback.Ty('x')


@pytest.mark.parametrize("module", [braided, balanced, ribbon],
                         ids=lambda m: m.__name__)
def test_b69_braid_dagger_roundtrip(module):
    braid = module.Braid(module.Ty('x'), module.Ty('y')).dagger()
    loaded = loads(dumps(braid))
    assert loaded == braid and loaded.is_dagger


def test_b69_delayed_box_roundtrip():
    box = feedback.Box('f', x, feedback.Ty('y')).d
    loaded = loads(dumps(box))
    assert loaded == box and loaded.time_step == 1


def test_b69_head_ob_roundtrip():
    assert loads(dumps(x.head)) == x.head


def test_b69_tail_ob_roundtrip():
    tail = feedback.Ty(feedback.Wire('x', is_constant=False)).tail
    assert loads(dumps(tail)) == tail


def test_b69_followed_by_roundtrip():
    fby = feedback.FollowedBy(x)
    assert loads(dumps(fby)) == fby
