"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Registered as a pytest plugin by the ``pytest11`` entry point, so the flag is
available wherever discopy is installed.

A doctest that needs an optional backend says so itself with a ``+EXTRA``
directive on one of its examples. What cannot say so -- a module that fails to
import, a test function, a notebook -- is caught by reading the error instead.
"""

import re

import pytest
from _pytest.doctest import DoctestItem


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


def needs_extra(item) -> bool:
    """ Whether a doctest declares itself as needing an optional backend. """
    return isinstance(item, DoctestItem) and item.dtest is not None and any(
        "+EXTRA" in example.source for example in item.dtest.examples)


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped before it runs. """
    if not config.getoption("--skip-extra"):
        return
    for item in items:
        if needs_extra(item):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """ A module that cannot be imported for want of a backend is skipped. """
    report = yield
    if not collector.config.getoption("--skip-extra"):
        return report
    if report.failed and (module := missing_module(str(report.longrepr))):
        report.outcome, report.longrepr = "skipped", (
            str(collector.path), None, f"Skipped: needs {module}")
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """ So is a test function or notebook that only fails for want of one. """
    try:
        return (yield)
    except BaseException as error:
        if not item.config.getoption("--skip-extra"):
            raise
        if (module := missing_dependency(error)) is None:
            raise
        pytest.skip(f"needs {module}")
