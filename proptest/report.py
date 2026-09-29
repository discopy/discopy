"""
What the property matrix actually checked, rather than that it was green.

A passing cell says the law held on the terms its strategy drew, which is
weaker than the law holding. Two things make a green matrix say less than a
reader takes it to say, and neither of them shows in a pass count:

- **a cell that stopped being checked.** A law skipped as inapplicable or
  declared broken is as green as one that held, so a change that quietly
  stops checking a law reads exactly like a change that keeps it. Learning
  that one merge had lost nothing cost two full matrix runs captured cell
  by cell and a diff of the two by hand.
- **a cell that cannot fail.** A strategy whose support is smaller than the
  budget draws the same terms however long the search runs, so the cell is
  green about those terms and silent about the law. On
  `#665 <https://github.com/discopy/discopy/pull/665>`_ ``serialisation``
  passed while ``dumps(Scalar(1j))`` raised, because the distribution held
  only ``Scalar(0.5)``.

Both are reports and neither is a gate: a law that becomes inapplicable and
a strategy with a finite support are legitimate, and failing the run for
either would only teach everyone to declare their way past it. This module
is the pure half — the outcome of a cell, the two readings and the JSON they
travel between runs in; :mod:`proptest.conftest` is the plugin that collects
one and prints the other.
"""

import json
from dataclasses import asdict, dataclass

PASSED, SKIPPED, XFAILED, XPASSED = (
    "passed", "skipped", "xfailed", "xpassed")

GONE = "gone"
"""The outcome of a cell the baseline had and the matrix no longer collects."""

STOPPED = (SKIPPED, XFAILED)
"""
The outcomes of a cell that is no longer checking its law: skipped because
the structure does not apply, or xfailed because the law is declared broken.
"""

CHECKED = (PASSED, XPASSED)
"""
The outcomes of a cell that did check its law. An ``xpassed`` cell checked
it and held — pytest reports the stale declaration itself — so losing one is
lost coverage like losing a passing cell, and gaining one is not a loss.
"""


@dataclass(frozen=True)
class Cell:
    """
    One cell of the matrix: its outcome, and how many distinct terms its
    strategy drew against the budget it was given.

    A cell that never reached its body has drawn nothing, so ``distinct``
    and ``budget`` are :obj:`None` rather than zero.

    >>> Cell("cat.Arrow.unitality", PASSED, distinct=20, budget=20)
    Cell(name='cat.Arrow.unitality', outcome='passed', distinct=20, budget=20)
    """
    name: str
    outcome: str
    distinct: int | None = None
    budget: int | None = None

    @property
    def exhausted(self):
        """
        Whether the cell drew fewer distinct terms than its budget allowed —
        a finite support, a filter rejecting most of one, or a search that
        stopped early. Whichever it is, the terms this cell can fail on have
        run out, so a larger budget buys it nothing.

        >>> Cell("a", PASSED, distinct=5, budget=100).exhausted
        True
        >>> Cell("b", PASSED, distinct=100, budget=100).exhausted
        False
        >>> Cell("c", SKIPPED).exhausted
        False
        """
        if self.outcome != PASSED or self.distinct is None:
            return False
        return self.distinct < self.budget


def classify(status, wasxfail=False):
    """
    The outcome of a cell, from the status pytest gives one of its phases
    and whether that phase carried an expected failure.

    >>> classify("skipped"), classify("skipped", wasxfail=True)
    ('skipped', 'xfailed')
    >>> classify("passed"), classify("passed", wasxfail=True)
    ('passed', 'xpassed')
    """
    if wasxfail:
        return XPASSED if status == PASSED else XFAILED
    return status


def exhausted(cells):
    """
    The passing cells whose strategy ran out of terms, worst first, so that
    the reader sees the emptiest support before the one that missed by one.

    >>> [cell.name for cell in exhausted([
    ...     Cell("a", PASSED, distinct=100, budget=100),
    ...     Cell("b", PASSED, distinct=20, budget=100),
    ...     Cell("c", PASSED, distinct=5, budget=100)])]
    ['c', 'b']
    """
    return tuple(sorted(
        (cell for cell in cells if cell.exhausted),
        key=lambda cell: (cell.distinct, cell.name)))


def stopped(baseline, cells):
    """
    The cells a baseline run checked and this one did not: checked there —
    passing, or xpassed and so checked all the same — and here skipped,
    declared broken, or gone from the matrix altogether.

    A cell that fails is not one of these — a failure reports itself — and
    neither is a cell the baseline had never checked.

    >>> was = [Cell("a", PASSED), Cell("b", PASSED), Cell("c", SKIPPED)]
    >>> now = [Cell("a", SKIPPED), Cell("c", PASSED)]
    >>> [(cell.name, cell.outcome) for cell in stopped(was, now)]
    [('a', 'skipped'), ('b', 'gone')]
    >>> [cell.outcome for cell in stopped([Cell("d", XPASSED)], [])]
    ['gone']
    """
    here = {cell.name: cell for cell in cells}
    checked = sorted(
        (cell.name for cell in baseline if cell.outcome in CHECKED))
    now = (here.get(name, Cell(name, GONE)) for name in checked)
    return tuple(cell for cell in now if cell.outcome in STOPPED + (GONE, ))


def render(cells, baseline=None):
    """
    The lines the run prints about itself: the cells that cannot fail, and,
    against a baseline, the cells that stopped being checked. An empty
    reading says so rather than printing nothing, since a section that
    vanishes when it is clean is indistinguishable from one nobody ran.

    ``baseline`` is :obj:`None` when the run was given no baseline and empty
    when it was given one with nothing in it, which is what an unreadable
    artifact comes to: the two read differently and only the first is the
    reader's to fix.

    >>> print("\\n".join(render([
    ...     Cell("cat.Arrow.identity_typing", PASSED, distinct=5, budget=100),
    ...     Cell("cat.Arrow.unitality", PASSED, distinct=100, budget=100)])))
    1 of 2 passing cell(s) drew fewer distinct terms than the budget allowed:
      cat.Arrow.identity_typing: 5 distinct term(s) of 100 examples
    no baseline to compare against: pass --proptest-baseline to name one
    """
    passing = [cell for cell in cells if cell.outcome == PASSED]
    lines = []
    if not cells:
        lines.append("no cell of the matrix was collected at all")
    elif not (out_of_terms := exhausted(cells)):
        lines.append(
            f"every one of the {len(passing)} passing cell(s) drew a distinct "
            "term for each example it was given")
    else:
        lines.append(
            f"{len(out_of_terms)} of {len(passing)} passing cell(s) drew "
            "fewer distinct terms than the budget allowed:")
        lines += [f"  {cell.name}: {cell.distinct} distinct term(s) "
                  f"of {cell.budget} examples" for cell in out_of_terms]
    if baseline is None:
        lines.append("no baseline to compare against: "
                     "pass --proptest-baseline to name one")
    elif not baseline:
        lines.append("the baseline names no cell to compare against")
    elif not (lost := stopped(baseline, cells)):
        lines.append(f"every cell the baseline checked is still checked "
                     f"({len(baseline)} cell(s) compared)")
    else:
        lines.append(f"{len(lost)} cell(s) the baseline checked and this run "
                     "did not:")
        lines += [f"  {cell.name}: {cell.outcome}" for cell in lost]
    return lines


def dumps(cells):
    """
    The report as JSON, sorted by cell so that two runs diff line by line.

    >>> print(dumps([Cell("a", PASSED, distinct=1, budget=2)]))
    {
      "a": {
        "budget": 2,
        "distinct": 1,
        "outcome": "passed"
      }
    }
    """
    return json.dumps(
        {cell.name: {key: value for key, value in asdict(cell).items()
                     if key != "name"} for cell in cells},
        indent=2, sort_keys=True)


def loads(source):
    """
    Read a report back, dropping a cell whose entry is not a mapping of the
    fields :class:`Cell` writes, and raising :class:`ValueError` on a
    top-level value that is not an object at all: the file is an artifact a
    previous run uploaded, so a baseline we cannot parse must be *reported*
    by the suite that reads it rather than crash it, and `json` raises
    nothing at all for a valid `[]` or `null`.

    >>> loads(dumps([Cell("a", PASSED, distinct=1, budget=2)]))
    (Cell(name='a', outcome='passed', distinct=1, budget=2),)
    >>> loads('{"a": "passed", "b": {"outcome": "passed"}}')
    (Cell(name='b', outcome='passed', distinct=None, budget=None),)
    >>> loads("[]")
    Traceback (most recent call last):
     ...
    ValueError: a report is an object keyed by cell, not a list
    """
    if not isinstance(entries := json.loads(source), dict):
        raise ValueError(f"a report is an object keyed by cell, not a "
                         f"{type(entries).__name__}")
    return tuple(
        Cell(name, entry["outcome"],
             distinct=entry.get("distinct"), budget=entry.get("budget"))
        for name, entry in sorted(entries.items())
        if isinstance(entry, dict) and "outcome" in entry)
