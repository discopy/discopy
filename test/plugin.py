"""
The ``--skip-extra`` flag, see CONTRIBUTING.md.

Everything that can say for itself that it needs an optional backend does:
a test with ``pytest.importorskip``, a doctest with a ``+EXTRA`` directive.
What is left cannot -- a module whose import is the thing that fails -- so
it is named here.
"""

import pytest
from _pytest.doctest import DoctestItem


UNIMPORTABLE = (
    "discopy/quantum/pennylane.py", "discopy/quantum/tk.py",
    "discopy/neural/torch.py", "discopy/neural/jax.py",
    "discopy/neural/cells.py", "discopy/neural/model.py",
    "discopy/neural/solver.py")


def pytest_addoption(parser):
    parser.addoption("--skip-extra", action="store_true", help=(
        "Skip what needs a dependency outside `uv sync --dev`, rather than "
        "fail. Nothing is skipped once the extras are installed."))


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--skip-extra"):
        return None
    return collection_path.as_posix().endswith(UNIMPORTABLE) or None


def pytest_collection_modifyitems(config, items):
    """ A doctest marked ``+EXTRA`` is skipped. """
    if not config.getoption("--skip-extra"):
        return
    for item in items:
        if isinstance(item, DoctestItem) and item.dtest is not None and any(
                "+EXTRA" in e.source for e in item.dtest.examples):
            item.add_marker(pytest.mark.skip(reason="needs an extra"))
