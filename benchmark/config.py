# -*- coding: utf-8 -*-

""" Shared configuration for declarative ``pytest-benchmark`` cases. """

import os
import time
from functools import wraps
from inspect import signature

import pytest


_FULL = "bench:full" in os.environ.get("BENCH_FLAGS", "").lower()


def sizes(*base, full=()):
    """ Sizes for a case: always ``base``, with the heavy ``full`` tail under
    ``BENCH_FLAGS=bench:full``. """
    return list(base) + (list(full) if _FULL else [])


def case(family, case, sizes, *, suite):
    """Benchmark metadata and shared measurement configuration."""
    benchmark = pytest.mark.benchmark(
        group=(suite, family, case), timer=time.process_time,
        disable_gc=True, max_time=.1, min_rounds=3)
    parametrize = pytest.mark.parametrize("n", sizes)

    def decorator(function):
        @wraps(function)
        def wrapped(benchmark, n):
            benchmark(function(n))

        wrapped.__signature__ = signature(wrapped, follow_wrapped=False)
        return benchmark(parametrize(wrapped))

    return decorator
