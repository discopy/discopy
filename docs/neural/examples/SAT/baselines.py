# -*- coding: utf-8 -*-

"""
The classical side of the study: an exact checker, a complete solver and the
two local-search baselines a learned model is scored against.

Nothing here knows about diagrams, torch or ``discopy``.  A formula is a
pair ``(n, clauses)`` with ``clauses`` a tuple of tuples of DIMACS literals
-- ``v + 1`` for the positive literal of variable ``v`` and ``-(v + 1)`` for
its negation -- and an assignment is a tuple or array of ``n`` booleans.

Four things, and they play three different parts:

* :func:`satisfies` and :func:`unsatisfied` -- the **exact verifier**.
  Counting satisfied clauses is what makes this domain kinder than sudoku:
  a decode is checked for free and without a learned critic.
* :func:`dpll` -- a complete solver, used as the **satisfiability filter**
  of the training set.  Small random instances are decided in milliseconds,
  and a filter has to be complete or the surviving distribution is biased
  towards whatever an incomplete solver happens to find.
* :func:`walksat` and :func:`gsat` -- the **baselines**, and also the fast
  path of the filter: a solution found is a certificate, so an instance
  ``walksat`` solves needs no complete search.  :func:`solve_within` runs
  either of them under a wall-clock budget, which is how the honest
  comparison of ``README.md`` is made.

Example
-------
>>> formula = ((1, 2, 3), (-1, 2, 3), (1, -2, 3))
>>> satisfies(3, formula, (False, False, True))
True
>>> unsatisfied(3, formula, (False, False, False))
1
>>> dpll(3, formula + ((1, 2, -3), (-1, -2, -3)))
True
>>> dpll(1, ((1, 1, 1), (-1, -1, -1)))
False
"""

from __future__ import annotations

import random
import time

import numpy as np


def code(literal: int) -> int:
    """
    A DIMACS literal as an index into a per-literal table: ``2 * v`` for the
    positive literal of variable ``v``, ``2 * v + 1`` for its negation.

    Parameters:
        literal : The DIMACS literal.

    Example
    -------
    >>> [code(literal) for literal in (1, -1, 2, -2)]
    [0, 1, 2, 3]
    """
    return 2 * (literal - 1) if literal > 0 else 2 * (-literal - 1) + 1


def occurrences(n: int, clauses) -> list[list[int]]:
    """
    The clauses each literal occurs in, indexed by :func:`code`.

    Parameters:
        n : The number of variables.
        clauses : The clauses, as tuples of DIMACS literals.
    """
    result: list[list[int]] = [[] for _ in range(2 * n)]
    for index, clause in enumerate(clauses):
        for literal in clause:
            result[code(literal)].append(index)
    return result


# --- the exact verifier ----------------------------------------------------

def satisfied_mask(clauses, assignment) -> np.ndarray:
    """
    Whether each clause is satisfied, as a boolean array.

    Parameters:
        clauses : The clauses, as an ``(m, k)`` array or a tuple of tuples.
        assignment : The truth value of each variable.
    """
    literals = np.asarray(clauses)
    values = np.asarray(assignment, dtype=bool)
    holds = values[np.abs(literals) - 1] == (literals > 0)
    return holds.any(-1)


def unsatisfied(n: int, clauses, assignment) -> int:
    """ The number of clauses an assignment leaves unsatisfied. """
    del n
    return int((~satisfied_mask(clauses, assignment)).sum())


def satisfies(n: int, clauses, assignment) -> bool:
    """ Whether an assignment satisfies every clause. """
    return unsatisfied(n, clauses, assignment) == 0


# --- a complete solver, i.e. the satisfiability filter ---------------------

def dpll(n: int, clauses, budget: int = 2_000_000) -> bool:
    """
    Whether a formula is satisfiable, by counter-based DPLL with unit
    propagation and a static Jeroslow-Wang decision heuristic.

    Iterative rather than recursive, so a deep search cannot overflow the
    Python stack, and complete: it answers ``True`` or ``False``, never
    "did not find one".  ``budget`` bounds the decisions taken and raises
    when exhausted, which at these sizes never happens and would be a
    silent bias if it were swallowed.

    Parameters:
        n : The number of variables.
        clauses : The clauses, as tuples of DIMACS literals.
        budget : The maximum number of decisions.

    Raises:
        RuntimeError : If the decision budget is exhausted.
    """
    occ = occurrences(n, clauses)
    n_true = [0] * len(clauses)
    n_open = [len(clause) for clause in clauses]
    value = [0] * (n + 1)
    trail: list[int] = []

    score = [0.0] * (2 * n)
    for clause in clauses:
        weight = 2.0 ** -len(clause)
        for literal in clause:
            score[code(literal)] += weight

    def assign(variable: int, truth: int, units: list) -> bool:
        # every occurrence is counted even once a conflict is known, so
        # that `undo` reverses exactly what was done.
        value[variable] = truth
        trail.append(variable)
        for index in occ[code(truth * variable)]:
            n_true[index] += 1
        consistent = True
        for index in occ[code(-truth * variable)]:
            n_open[index] -= 1
            if not n_true[index]:
                if not n_open[index]:
                    consistent = False
                elif n_open[index] == 1:
                    units.append(index)
        return consistent

    def propagate(units: list) -> bool:
        while units:
            index = units.pop()
            if n_true[index] or n_open[index] != 1:
                continue
            free = next(literal for literal in clauses[index]
                        if not value[abs(literal)])
            if not assign(abs(free), 1 if free > 0 else -1, units):
                return False
        return True

    def undo(mark: int) -> None:
        while len(trail) > mark:
            variable = trail.pop()
            truth, value[variable] = value[variable], 0
            for index in occ[code(truth * variable)]:
                n_true[index] -= 1
            for index in occ[code(-truth * variable)]:
                n_open[index] += 1

    def decide() -> int:
        best, chosen = -1.0, 0
        for variable in range(1, n + 1):
            if value[variable]:
                continue
            positive = score[2 * variable - 2]
            negative = score[2 * variable - 1]
            if positive + negative > best:
                best, chosen = positive + negative, (
                    variable if positive >= negative else -variable)
        return chosen

    if any(not length for length in n_open):
        return False
    if not propagate([index for index, length in enumerate(n_open)
                      if length == 1]):
        return False

    stack: list[tuple[int, int, bool]] = []
    pending, flipped = decide(), False
    if not pending:
        return True
    for _ in range(budget):
        mark, units = len(trail), []
        if assign(abs(pending), 1 if pending > 0 else -1, units) \
                and propagate(units):
            stack.append((mark, pending, flipped))
            pending, flipped = decide(), False
            if not pending:
                return True
            continue
        undo(mark)
        if not flipped:
            pending, flipped = -pending, True
            continue
        while stack:
            mark, literal, was_flipped = stack.pop()
            undo(mark)
            if not was_flipped:
                pending, flipped = -literal, True
                break
        else:
            return False
    raise RuntimeError(f"dpll exhausted {budget} decisions")


# --- the local-search baselines -------------------------------------------

def random_assignment(n: int, rng: random.Random) -> list[bool]:
    """ A uniform random assignment: the trivial baseline. """
    return [rng.random() < 0.5 for _ in range(n)]


class Search:
    """
    The incremental bookkeeping :func:`walksat` and :func:`greedy` share:
    the truth value of every variable, how many literals of each clause are
    true, and the list of currently unsatisfied clauses.

    Keeping the count of true literals per clause is what makes a flip cost
    ``O(degree)`` rather than ``O(m)``: a flip breaks exactly the clauses in
    which the flipped literal was the only true one.

    Parameters:
        n : The number of variables.
        clauses : The clauses, as tuples of DIMACS literals.
        assignment : The starting assignment.

    Example
    -------
    >>> search = Search(3, ((1, 2, 3), (-1, -2, -3)), (True, True, True))
    >>> search.unsat, search.breaks(1)
    ([1], 0)
    >>> search.flip(1)
    >>> search.unsat, search.value
    ([], [False, True, True])
    """
    def __init__(self, n: int, clauses, assignment):
        self.n, self.clauses = n, tuple(clauses)
        self.occ = occurrences(n, clauses)
        self.value = list(assignment)
        self.n_true = [0] * len(self.clauses)
        self.unsat: list[int] = []
        self.where = [-1] * len(self.clauses)
        for index, clause in enumerate(self.clauses):
            for literal in clause:
                if self.value[abs(literal) - 1] == (literal > 0):
                    self.n_true[index] += 1
            if not self.n_true[index]:
                self.where[index] = len(self.unsat)
                self.unsat.append(index)

    def breaks(self, variable: int) -> int:
        """ How many satisfied clauses flipping a variable would break. """
        literal = variable if self.value[variable - 1] else -variable
        return sum(1 for index in self.occ[code(literal)]
                   if self.n_true[index] == 1)

    def flip(self, variable: int) -> None:
        """ Flip a variable and repair the bookkeeping around it. """
        was = self.value[variable - 1]
        self.value[variable - 1] = not was
        for index in self.occ[code(variable if was else -variable)]:
            self.n_true[index] -= 1
            if not self.n_true[index]:
                self.where[index] = len(self.unsat)
                self.unsat.append(index)
        for index in self.occ[code(-variable if was else variable)]:
            self.n_true[index] += 1
            if self.n_true[index] == 1:
                position, last = self.where[index], self.unsat.pop()
                if last != index:
                    self.unsat[position] = last
                    self.where[last] = position
                self.where[index] = -1


def walksat(n: int, clauses, max_flips: int = 20000, tries: int = 10,
            noise: float = 0.5, rng: random.Random = None,
            deadline: float = None):
    """
    WalkSAT/SKC (Selman, Kautz & Cohen 1994): pick an unsatisfied clause at
    random; if one of its variables can be flipped without breaking any
    satisfied clause, flip it, otherwise flip a random one with probability
    ``noise`` and the least-breaking one else.

    Incomplete by construction: a returned assignment is a certificate that
    the formula is satisfiable, but ``None`` means only that the budget ran
    out.

    Parameters:
        n : The number of variables.
        clauses : The clauses, as tuples of DIMACS literals.
        max_flips : The flips per try.
        tries : The random restarts.
        noise : The probability of a random walk step.
        rng : The random generator, a fresh one by default.
        deadline : A ``time.perf_counter`` value to stop at.

    Returns:
        A satisfying assignment, or ``None``.
    """
    rng = random.Random() if rng is None else rng
    for _ in range(tries):
        search = Search(n, clauses, random_assignment(n, rng))
        for flip in range(max_flips):
            if not search.unsat:
                return search.value
            if deadline is not None and not flip % 256 \
                    and time.perf_counter() > deadline:
                return None
            clause = search.clauses[rng.choice(search.unsat)]
            variables = [abs(literal) for literal in clause]
            costs = [search.breaks(variable) for variable in variables]
            best = min(costs)
            if best and rng.random() < noise:
                search.flip(rng.choice(variables))
            else:
                search.flip(variables[costs.index(best)])
        if not search.unsat:
            return search.value
    return None


def greedy(n: int, clauses, max_flips: int = 20000, tries: int = 10,
           rng: random.Random = None, deadline: float = None):
    """
    Greedy descent: from a random assignment, repeatedly flip the variable
    of an unsatisfied clause that breaks the fewest satisfied clauses,
    restarting once it has spent ``4 * n`` flips without a free one.

    This is the greedy baseline the success criterion of Part 1 is stated
    against, and it is exactly :func:`walksat` with ``noise=0``, so the
    difference between the two lines is the random-walk step and nothing
    else.

    Parameters:
        n, clauses, max_flips, tries, rng, deadline : As in :func:`walksat`.

    Returns:
        A satisfying assignment, or ``None``.
    """
    rng = random.Random() if rng is None else rng
    for _ in range(tries):
        search = Search(n, clauses, random_assignment(n, rng))
        stalled = 0
        for flip in range(max_flips):
            if not search.unsat:
                return search.value
            if deadline is not None and not flip % 256 \
                    and time.perf_counter() > deadline:
                return None
            clause = search.clauses[rng.choice(search.unsat)]
            variables = [abs(literal) for literal in clause]
            costs = [search.breaks(variable) for variable in variables]
            best = min(costs)
            stalled = 0 if best == 0 else stalled + 1
            if stalled > 4 * n:
                break
            search.flip(variables[costs.index(best)])
    return None


SOLVERS = {"walksat": walksat, "greedy": greedy}


def solve_within(name: str, n: int, clauses, seconds: float,
                 rng: random.Random = None, **kwargs) -> bool:
    """
    Whether a local-search baseline solves a formula within a wall-clock
    budget, which is how a learned model and a classical one are compared
    at matched compute.

    Parameters:
        name : ``"walksat"`` or ``"greedy"``.
        n : The number of variables.
        clauses : The clauses.
        seconds : The wall-clock budget.
        rng : The random generator.
        kwargs : Passed to the solver, e.g. ``noise``.
    """
    return SOLVERS[name](
        n, clauses, rng=rng, tries=10 ** 9, max_flips=10 ** 9,
        deadline=time.perf_counter() + seconds, **kwargs) is not None


def satisfiable(n: int, clauses, rng: random.Random = None,
                flips: int = 4000, tries: int = 4) -> bool:
    """
    Whether a formula is satisfiable, deciding it completely.

    :func:`walksat` runs first because a solution it finds is a certificate
    and most instances away from the threshold fall to it in milliseconds;
    :func:`dpll` decides the rest.  The answer is therefore exact, which a
    filter has to be: keeping "whatever local search could solve" would
    quietly remove the hard satisfiable instances from the training set.

    Parameters:
        n : The number of variables.
        clauses : The clauses.
        rng : The random generator behind the local search.
        flips : The flips per WalkSAT try.
        tries : The WalkSAT restarts before falling back to DPLL.
    """
    if walksat(n, clauses, max_flips=flips, tries=tries, rng=rng) is not None:
        return True
    return dpll(n, clauses)
