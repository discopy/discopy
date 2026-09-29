"""
Unit tests for :mod:`proptest.report`, the suite's reading of itself.

The doctests state what each function does on one example; these state the
cases a reader would otherwise have to trust: what counts as a cell that
stopped being checked, what an unreadable baseline comes to, and that the
JSON is a round trip.
"""

import pytest

from proptest.conftest import cell_name
from proptest.report import (
    GONE, PASSED, SKIPPED, XFAILED, XPASSED,
    Cell, classify, dumps, exhausted, loads, render, stopped)


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
