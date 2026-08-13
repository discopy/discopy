"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.
"""

import pytest
from _pytest.doctest import DoctestItem


UNIMPORTABLE = ("discopy/quantum/pennylane.py", "discopy/quantum/tk.py")


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))
    parser.addoption("--bugs", choices=("xfail", "skip"), default="xfail",
                     help="Whether axioms marked bug are xfailed or skipped.")


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--skip-extra"):
        return None
    return collection_path.as_posix().endswith(UNIMPORTABLE) or None


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped. """
    for item in items:
        if config.getoption("--skip-extra") and isinstance(
                item, DoctestItem) and item.dtest is not None and any(
                "+EXTRA" in e.source for e in item.dtest.examples):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))
        axiom = getattr(getattr(item, "callspec", None), "params", {}).get(
            "axiom")
        if config.getoption("--bugs") == "skip" and getattr(
                axiom, "status", None) == "bug":
            item.add_marker(pytest.mark.skip(reason=f"{axiom.name} is a bug"))
