"""
Unit tests for :mod:`proptest.report`, the suite's reading of itself.

The doctests state what each function does on one example; these state the
cases a reader would otherwise have to trust: what counts as a cell that
stopped being checked, what an unreadable baseline comes to, and that the
JSON is a round trip.
"""

import pytest

import proptest.conftest as plugin
from proptest.conftest import cell_name
from proptest.report import (
    GONE, PASSED, SKIPPED, UNSETTLED, XFAILED, XPASSED,
    Cell, classify, dumps, exhausted, loads, render, stopped)

CELL = "proptest/test_axioms.py::test_axiom[cat.Arrow.unitality]"


class Phase:
    """ One phase of one cell, as pytest hands it to the plugin. """
    def __init__(self, when, outcome, properties=(), wasxfail=False):
        self.when, self.outcome, self.nodeid = when, outcome, CELL
        self.user_properties = list(properties)
        self.keywords = {plugin.CELL: 1}
        if wasxfail:
            self.wasxfail = ""


@pytest.fixture
def collected(monkeypatch):
    """ Drive the plugin's collector over a fresh, isolated dictionary. """
    monkeypatch.setattr(plugin, "CELLS", {})

    def run(*phases):
        for phase in phases:
            plugin.pytest_runtest_logreport(phase)
        return plugin.CELLS["cat.Arrow.unitality"]
    return run


def test_a_cell_whose_body_never_ran_is_not_passing(collected):
    """
    Setup passing is not the law holding. A collector seeded with `passed`
    reported a cell interrupted between setup and call as one that held on a
    distinct term per example, which is the defect this module reports on.
    """
    assert collected(Phase("setup", PASSED)).outcome == UNSETTLED


def test_the_call_phase_settles_a_cell(collected):
    """ And teardown carries the numbers `drawn` recorded. """
    cell = collected(
        Phase("setup", PASSED), Phase("call", PASSED),
        Phase("teardown", PASSED,
              [(plugin.DISTINCT, 20), (plugin.BUDGET, 20)]))
    assert (cell.outcome, cell.distinct, cell.budget) == (PASSED, 20, 20)


def test_a_skip_at_setup_settles_a_cell(collected):
    """ A cell the matrix declares inapplicable never reaches a call. """
    assert collected(Phase("setup", SKIPPED)).outcome == SKIPPED


def test_a_declared_failure_settles_a_cell(collected):
    """ An xfail is a skip at call carrying `wasxfail`. """
    assert collected(
        Phase("setup", PASSED),
        Phase("call", SKIPPED, wasxfail=True)).outcome == XFAILED


def test_an_unsettled_cell_is_reported_in_its_own_right():
    """ A run that did not finish says so before anything else. """
    assert render([Cell("a", UNSETTLED)])[:2] == [
        "1 cell(s) started and never finished, so this run says nothing "
        "about their law(s):",
        "  a"]


def test_losing_a_cell_to_an_unsettled_run_is_lost_coverage():
    """ A law the baseline checked and this run never reached. """
    lost, = stopped([Cell("a", PASSED)], [Cell("a", UNSETTLED)])
    assert lost.outcome == UNSETTLED


def test_classify_reads_an_expected_failure():
    """ A phase's status is a cell's outcome unless it was declared broken. """
    assert classify("failed") == "failed"
    assert classify(SKIPPED) == SKIPPED
    assert classify(SKIPPED, wasxfail=True) == XFAILED
    assert classify(PASSED, wasxfail=True) == XPASSED


@pytest.mark.parametrize("outcome", [SKIPPED, XFAILED, XPASSED, "failed"])
def test_only_a_passing_cell_can_be_exhausted(outcome):
    """
    A cell that did not pass is reported by pytest itself, whatever its
    strategy managed to draw, so it is never one of these.
    """
    assert not exhausted([Cell("a", outcome, distinct=1, budget=100)])


def test_a_cell_that_drew_nothing_is_not_exhausted():
    """
    A skipped cell never reached its body, so it drew no terms rather than
    too few: there is nothing to report about a law nobody stated.
    """
    assert not exhausted([Cell("a", SKIPPED), Cell("b", PASSED)])


def test_exhausted_is_worst_first():
    """ The emptiest support is the one worth reading first. """
    cells = [Cell("a", PASSED, distinct=99, budget=100),
             Cell("b", PASSED, distinct=1, budget=100),
             Cell("c", PASSED, distinct=100, budget=100)]
    assert [cell.name for cell in exhausted(cells)] == ["b", "a"]


@pytest.mark.parametrize("outcome", [SKIPPED, XFAILED])
def test_a_cell_the_baseline_checked_and_this_run_does_not(outcome):
    """ Skipped and declared broken are both no longer checking the law. """
    lost, = stopped([Cell("a", PASSED)], [Cell("a", outcome)])
    assert (lost.name, lost.outcome) == ("a", outcome)


def test_a_cell_the_matrix_no_longer_collects_is_gone():
    """ A law nobody states any more is the same loss as a skipped one. """
    lost, = stopped([Cell("a", PASSED)], [])
    assert (lost.name, lost.outcome) == ("a", GONE)


@pytest.mark.parametrize("outcome", ["failed", XPASSED])
def test_a_failure_and_a_stale_declaration_are_not_lost_coverage(outcome):
    """
    Both are reported by pytest, loudly, and both did check their law: a
    second line about them would be noise rather than a reading.
    """
    assert not stopped([Cell("a", PASSED)], [Cell("a", outcome)])


@pytest.mark.parametrize("outcome", [SKIPPED, XFAILED, GONE])
def test_losing_an_xpassed_baseline_cell_is_lost_coverage(outcome):
    """
    An `xpassed` cell checked its law and the law held; pytest reports the
    stale declaration, not the coverage. So a run that stops checking it has
    lost exactly what a lost passing cell loses.
    """
    now = [] if outcome == GONE else [Cell("a", outcome)]
    lost, = stopped([Cell("a", XPASSED)], now)
    assert (lost.name, lost.outcome) == ("a", outcome)


def test_a_run_that_collected_no_cell_says_so():
    """
    A matrix that collects nothing is the largest loss there is, and a
    section that disappears rather than saying so is the defect this module
    exists to remove.
    """
    assert render([], baseline=[Cell("a", PASSED)]) == [
        "no cell of the matrix was collected at all",
        "1 cell(s) the baseline checked and this run did not:",
        "  a: gone"]


def test_a_cell_the_baseline_had_not_checked_either_is_not_a_loss():
    """
    A law that was already skipped and still is has lost nothing, and one
    this run added is a gain nobody needs warning about.
    """
    assert not stopped(
        [Cell("a", SKIPPED)], [Cell("a", SKIPPED), Cell("b", PASSED)])


def test_render_says_so_when_there_is_nothing_to_say():
    """
    A clean reading prints its own line: a section that disappears when it
    is clean cannot be told from one that never ran.
    """
    cells = [Cell("a", PASSED, distinct=20, budget=20)]
    assert render(cells, baseline=cells) == [
        "every one of the 1 passing cell(s) drew a distinct term for each "
        "example it was given",
        "every cell the baseline checked is still checked (1 cell(s) "
        "compared)"]


def test_render_tells_no_baseline_from_an_empty_one():
    """
    No baseline is the reader's to fix by naming one; an empty baseline is
    what an unreadable artifact comes to and names itself.
    """
    cells = [Cell("a", PASSED, distinct=20, budget=20)]
    assert render(cells)[-1].startswith("no baseline")
    assert render(cells, baseline=())[-1] == (
        "the baseline names no cell to compare against")


@pytest.mark.parametrize("cell", [
    Cell("a", PASSED, distinct=5), Cell("a", PASSED, budget=20),
    Cell("a", PASSED)])
def test_a_passing_cell_with_a_missing_count_is_not_exhausted(cell):
    """
    And comparing the one it has against the one it has not used to raise
    `TypeError: '<' not supported between instances of 'int' and 'NoneType'`.
    """
    assert not cell.exhausted
    assert not exhausted([cell])


def test_a_passing_cell_with_no_count_is_reported_as_unknown():
    """
    Not as one that drew a distinct term per example, which is what the
    all-distinct line used to claim about it. A cell whose test does not take
    the `drawn` fixture is the reachable case.
    """
    assert render([Cell("a", PASSED), Cell("b", PASSED, distinct=9, budget=9)],
                  baseline=())[:2] == [
        "1 of 2 passing cell(s) recorded no distinct-term count, so this run "
        "does not say how hard their law(s) were tried:",
        "  a"]


def test_the_all_distinct_line_needs_every_count():
    """ A claim about every passing cell, so one unknown withholds it. """
    counted = [Cell("b", PASSED, distinct=9, budget=9)]
    assert render(counted, baseline=())[0].startswith("every one of the 1")
    assert not any(line.startswith("every one of")
                   for line in render(counted + [Cell("a", PASSED)],
                                      baseline=()))


def test_the_report_is_a_round_trip():
    """ What one run writes is what the next one reads. """
    cells = (Cell("a", PASSED, distinct=5, budget=100), Cell("b", SKIPPED))
    assert loads(dumps(cells)) == cells


def test_an_entry_that_is_not_a_cell_is_dropped():
    """
    The baseline is an artifact of an older run, so a reader of it keeps the
    cells it understands rather than failing the suite that asked for it.
    """
    assert loads('{"a": null, "b": {}, "c": {"outcome": "passed"}}') == (
        Cell("c", PASSED), )


@pytest.mark.parametrize("source", ["[]", "null", '"passed"', "3"])
def test_a_report_that_is_not_an_object_raises_ValueError(source):
    """
    `json` raises nothing for a valid `[]` or `null`, and the caller catches
    `OSError` and `ValueError` only: without this the unreadable-baseline
    diagnostic the docstring promises would be an `AttributeError` that takes
    the whole suite down.
    """
    with pytest.raises(ValueError):
        loads(source)


def test_cell_name_is_the_parameter_pytest_prints():
    """ A node with no parameter is not a cell of the matrix. """
    assert cell_name("proptest/test_axioms.py::test_axiom[cat.Arrow.x]") == (
        "cat.Arrow.x")
    assert cell_name("proptest/test_report.py::test_the_report") is None
    assert cell_name("proptest/report.py::proptest.report.Cell") is None
