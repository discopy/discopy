"""
Hypothesis profiles, example databases and the run's own report for the
property suite, see the documentation of :mod:`discopy.axioms` and of
:mod:`proptest.report`.

Under ``CI`` a registered profile inherits Hypothesis's ``ci`` defaults,
``derandomize=True`` and hence ``database=None``, so both are explicit.
"""

import os
import pathlib

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase, GitHubArtifactDatabase,
    MultiplexedDatabase, ReadOnlyDatabase)

from proptest.report import (
    PASSED, Cell, classify, dumps, loads, render)

LOCAL = DirectoryBasedExampleDatabase(".hypothesis/examples")
"""
The database every run writes to: on CI it is downloaded from the
previous run's artifact before the tests and uploaded after them.
"""

COMMON = dict(
    derandomize=False, database=LOCAL, deadline=None, print_blob=True,
    suppress_health_check=[
        HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
"""
The settings every profile shares. The function-scoped fixture Hypothesis
warns about is :func:`drawn`, which counts what a cell drew across the whole
search and is meant to outlive one example.
"""


PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")

settings.register_profile("pr", max_examples=20, **COMMON)
settings.register_profile("explore", max_examples=1000, **COMMON)
settings.register_profile("dev", max_examples=100, **COMMON)
if PROFILE != "shared":
    settings.load_profile(PROFILE)


def pytest_configure(config):
    """
    Register and load the ``shared`` profile on demand: the ``dev`` budget
    over the local database backed by CI's, read-only, so that a developer
    with a ``GITHUB_TOKEN`` replays what CI found without recording
    anything. Building the artifact database touches storage, which
    Hypothesis warns against at conftest import, so it happens only when
    the profile is asked for.
    """
    if PROFILE == "shared":
        database = MultiplexedDatabase(LOCAL, ReadOnlyDatabase(
            GitHubArtifactDatabase("discopy", "discopy")))
        settings.register_profile(
            "shared", max_examples=100, **dict(COMMON, database=database))
        settings.load_profile("shared")


CELLS = {}
"""
What this process has collected, keyed by cell. A worker under ``-n auto``
fills its own copy and the controller fills none of them, so the cells reach
the summary through the reports below rather than through this dictionary.
"""


@pytest.fixture
def drawn(record_property):
    """
    The distinct terms one cell draws, as a set the test adds to.

    Hypothesis sets a function-scoped fixture up once for the whole search
    rather than once per example, which is what this one wants: the count is
    over the cell, not over an example. It reaches the controller through
    ``record_property`` because a worker process is where it is collected and
    the summary is printed somewhere else.
    """
    terms = set()
    yield terms
    record_property(DISTINCT, len(terms))
    record_property(BUDGET, settings().max_examples)


DISTINCT, BUDGET = "proptest-distinct", "proptest-budget"
"""The names :func:`drawn` records its two numbers under."""


def cell_name(nodeid):
    """
    The name of the cell a node id belongs to, i.e. the parameter pytest
    prints in brackets, which :func:`proptest.categories.category_parameters`
    and ``test_axioms.axiom_parameters`` build out of the category and the
    law. A node with no parameter is not a cell and has no name.

    >>> cell_name("proptest/test_axioms.py::test_axiom[cat.Arrow.unitality]")
    'cat.Arrow.unitality'
    >>> cell_name("proptest/test_report.py::test_stopped")
    """
    _, bracket, parameter = nodeid.partition("[")
    return parameter[:-1] if bracket and parameter.endswith("]") else None


def pytest_addoption(parser):
    """ Where to write this run's report, and which one to compare against. """
    group = parser.getgroup("proptest")
    group.addoption(
        "--proptest-report", metavar="PATH", default=None,
        help="write this run's cell report to PATH, as JSON")
    group.addoption(
        "--proptest-baseline", metavar="PATH", default=None,
        help="compare this run against the report at PATH, naming the cells "
             "it checked and this run did not")


def pytest_runtest_logreport(report):
    """
    Collect each cell's outcome and the two numbers :func:`drawn` recorded.

    A cell reports up to three phases: the one that settles it is the call,
    or whichever phase did not pass — a skip at setup, for the cells the
    matrix declares inapplicable. Teardown is where the properties arrive,
    so the numbers are merged into what is already there.
    """
    if (name := cell_name(report.nodeid)) is None:
        return
    known = CELLS.get(name, Cell(name, PASSED))
    settled = report.when == "call" or report.outcome != PASSED
    properties = dict(report.user_properties)
    CELLS[name] = Cell(
        name,
        classify(report.outcome, hasattr(report, "wasxfail"))
        if settled else known.outcome,
        distinct=properties.get(DISTINCT, known.distinct),
        budget=properties.get(BUDGET, known.budget))


def pytest_terminal_summary(terminalreporter, config):
    """
    Print what the matrix checked, and write the report the next run reads
    as its baseline. A baseline that cannot be read is named and skipped: it
    is an artifact of an older run and must not fail the suite reading it.
    """
    cells = tuple(CELLS.values())
    if not cells:
        return
    terminalreporter.section("what the matrix checked")
    baseline, path = None, config.getoption("--proptest-baseline")
    if path is not None:
        try:
            baseline = loads(pathlib.Path(path).read_text())
        except (OSError, ValueError) as error:
            baseline = ()
            terminalreporter.write_line(f"unreadable baseline {path}: {error}")
    for line in render(cells, baseline):
        terminalreporter.write_line(line)
    if (path := config.getoption("--proptest-report")) is not None:
        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(dumps(cells))
        terminalreporter.write_line(f"report written to {path}")
