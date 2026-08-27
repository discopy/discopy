"""
The ``--skip-extra`` and ``--axioms`` flags, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.
"""

import re

import pytest
from _pytest.doctest import DoctestItem


UNIMPORTABLE = ("discopy/quantum/pennylane.py", "discopy/quantum/tk.py")


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))
    parser.addoption("--axioms", help=(
        "Select the parametrized tests whose id matches this glob, e.g. "
        "--axioms 'compact.CMap.*' or --axioms '*.Diagram.unitality'. "
        "Only `*` is a wildcard, so brackets match themselves, as in "
        "--axioms 'hypergraph.Hypergraph[compact.Diagram].*'."))


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--skip-extra"):
        return None
    return collection_path.as_posix().endswith(UNIMPORTABLE) or None


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped. """
    if pattern := config.getoption("--axioms"):
        glob = re.compile(".*".join(map(re.escape, pattern.split("*"))))
        selected, deselected = [], []
        for item in items:
            spec = getattr(getattr(item, "callspec", None), "id", "")
            (selected if glob.fullmatch(spec)
             else deselected).append(item)
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    if not config.getoption("--skip-extra"):
        return
    for item in items:
        if isinstance(item, DoctestItem) and item.dtest is not None and any(
                "+EXTRA" in e.source for e in item.dtest.examples):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))
