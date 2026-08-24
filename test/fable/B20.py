# -*- coding: utf-8 -*-
"""B20: Channel[float].discard and cups lose the dtype (discopy/quantum/channel.py:294).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.quantum.channel import Channel, Q
from discopy.tensor import Dim


def test_b20_discard_keeps_dtype():
    assert type(Channel[float].discard(Q(Dim(2)))) is Channel[float]


def test_b20_cups_keep_dtype():
    assert type(Channel[float].cups(Q(Dim(2)), Q(Dim(2)))) is Channel[float]
