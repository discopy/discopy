""" Skip what needs more than ``uv sync --dev``, see CONTRIBUTING.md. """

import re

import pytest


OPTIONAL = frozenset((
    "graphviz", "jax", "jaxlib", "nltk", "pennylane", "pytket", "pyzx",
    "qiskit", "quimb", "sympy", "tensornetwork", "torch"))


def causes(error: BaseException):
    """ An error, what it wraps (doctest nests them) and all their causes. """
    while error is not None:
        yield error
        for failure in getattr(error, "failures", ()):
            yield from causes(failure)
        if getattr(error, "exc_info", None) is not None:
            yield from causes(error.exc_info[1])
        error = error.__cause__ or error.__context__


def missing_module(text: str) -> str | None:
    """ The optional dependency that a message complains about, if any. """
    if "Graphviz executable" in text:
        return "graphviz"
    for module in re.findall(r"No module named '([\w.]+)'", text):
        if module.split(".")[0] in OPTIONAL:
            return module
    return None


def missing_dependency(error: BaseException) -> str | None:
    """ The same, for an exception and everything it wraps. """
    for cause in causes(error):
        if isinstance(cause, ModuleNotFoundError) and cause.name in OPTIONAL:
            return cause.name
        if module := missing_module(str(cause)):
            return module
    return None


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """ A module that cannot be imported for want of a backend is skipped. """
    report = yield
    if report.failed and (module := missing_module(str(report.longrepr))):
        report.outcome, report.longrepr = "skipped", (
            str(collector.path), None, f"Skipped: needs {module}")
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """ So is a test that only fails for want of one. """
    try:
        return (yield)
    except BaseException as error:
        if (module := missing_dependency(error)) is None:
            raise
        pytest.skip(f"needs {module}")
