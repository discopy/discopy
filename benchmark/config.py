# -*- coding: utf-8 -*-

""" Shared configuration for declarative ``pytest-benchmark`` cases. """

import os
import time

import pytest


_FULL = "bench:full" in os.environ.get("BENCH_FLAGS", "").lower()


def sizes(*base, full=()):
    """ Sizes for a case: always ``base``, with the heavy ``full`` tail under
    ``BENCH_FLAGS=bench:full``. """
    return list(base) + (list(full) if _FULL else [])


def case(group):
    """ Shared benchmark marker: CPU-time clock, garbage collector disabled
    in the timed region, grouped by case name. """
    return pytest.mark.benchmark(
        group=group, timer=time.process_time, disable_gc=True)


# Median of ROUNDS timed calls after WARMUP untimed ones. Inputs are built
# once outside the timed thunk, so only the operation under test is measured.
ROUNDS, WARMUP = 3, 1
