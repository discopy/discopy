"""
Turn a missing optional dependency into a skip, rather than an error.

The default install has only the dependencies of :mod:`discopy` itself, so a
module wrapping an optional backend cannot be imported at all. Two modules of
the package are in that case: they are listed in ``REQUIREMENTS`` and ignored
before collection. The test modules in the same case skip themselves with
``pytest.importorskip``, the only thing that works for a path named directly
in ``testpaths``.

Everywhere else the dependency is needed by one doctest rather than a whole
file, so we let the test run and read the error: if it is one of the
``OPTIONAL`` backends missing, the test is skipped rather than failed.

Install the extras (``uv sync --dev --group all``) and nothing is skipped.
"""

import re
from importlib.util import find_spec

import pytest


REQUIREMENTS = {
    "discopy/quantum/pennylane.py": "pennylane",
    "discopy/quantum/tk.py": "pytket",
}

OPTIONAL = frozenset((
    "jax", "jaxlib", "nltk", "pennylane", "pytket", "pyzx", "qiskit", "sympy",
    "tensornetwork", "torch", "quimb"))


def is_available(module: str) -> bool:
    """ Whether ``module`` can be imported, without importing it. """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


collect_ignore = [
    path for path, module in REQUIREMENTS.items() if not is_available(module)]


def causes(error: BaseException):
    """ An error, what it wraps (doctest nests them) and all their causes. """
    while error is not None:
        yield error
        for failure in getattr(error, "failures", ()):
            yield from causes(failure)
        if getattr(error, "exc_info", None) is not None:
            yield from causes(error.exc_info[1])
        error = error.__cause__ or error.__context__


def missing_dependency(error: BaseException) -> str | None:
    """
    The optional dependency that ``error`` is really about, if any.

    A notebook reports the error as text rather than as an exception, hence
    the regular expression as well as the :class:`ModuleNotFoundError`.
    """
    for cause in causes(error):
        if isinstance(cause, ModuleNotFoundError) and cause.name in OPTIONAL:
            return cause.name
        if "Graphviz executable" in str(cause):
            return "graphviz"
        for module in re.findall(r"No module named '([\w.]+)'", str(cause)):
            if module.split(".")[0] in OPTIONAL:
                return module
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """ Skip a test that only fails for want of an optional dependency. """
    try:
        return (yield)
    except BaseException as error:
        module = missing_dependency(error)
        if module is None:
            raise
        pytest.skip(f"needs {module}")
