# -*- coding: utf-8 -*-

"""
Paths, widths and budgets of the random-SAT study.

Everything here is a plain value: no torch, no diagrams, nothing that has to
be built.  A :class:`Widths` fixes the parameter count of a model, a
:class:`Budget` fixes what a run is allowed to spend, and
:data:`GRID` fixes the ``(n, alpha)`` points every protocol reports on.

Unlike the sudoku study, every *instance* here is its own diagram, so a
budget also has to say how many diagrams are compiled and how often they are
replaced; see :class:`Budget` and ``NOTES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The clause width of the benchmark: uniform random 3-SAT throughout.
K = 3

#: Everything this example reads and writes lives inside its own directory,
#: so the whole study is one relocatable folder.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

for _directory in (DATA_DIR, ARTIFACTS, FIGURES):
    _directory.mkdir(parents=True, exist_ok=True)

#: The clause-to-variable ratios swept, the last one being the satisfiability
#: threshold of random 3-SAT.
ALPHAS = (3.0, 3.5, 4.0, 4.26)

#: The variable counts of the training distribution, uniform over the range.
N_RANGE = (10, 100)

#: The variable counts of the held-out grid, one test set per ``(n, alpha)``.
GRID_N = (10, 25, 50, 75, 100)

#: The larger instances generated but *not* trained or tested on here: they
#: are Part 2's size-generalization study and are kept out of Part 1 so that
#: no protocol can quietly touch them.
BIG_N = (150, 200, 300)

#: The grid every protocol reports on.
GRID = tuple((n, alpha) for n in GRID_N for alpha in ALPHAS)

#: The seeds every model is trained with, in order.
SEEDS = (0, 1, 2)

#: Gradient-norm clipping, identical for every model.
GRAD_CLIP = 1.0


@dataclass(frozen=True)
class Widths:
    """
    The widths of one model, the knobs that set its parameter count.

    Parameters:
        dim : The width of a literal-to-relation message.
        state_dim : The width of a literal's recurrent state, and of a
                    clause's when the clause is stateful.
        hidden : The width of the hidden layers inside a cell.
    """
    dim: int = 32
    state_dim: int = 64
    hidden: int = 128

    def asdict(self) -> dict:
        return {"dim": self.dim, "state_dim": self.state_dim,
                "hidden": self.hidden}


@dataclass(frozen=True)
class Budget:
    """
    One experiment budget: how much data, how many diagrams, how long.

    A diagram costs a compilation, and a compilation is Python; a training
    step costs a fixed number of kernel launches whatever the batch holds
    (see ``NOTES.md``).  So a budget spends its time on **pool_batches**
    compiled batches, reuses each of them for ``refresh_every`` epochs, and
    then throws the pool away and compiles a fresh one -- which is the knob
    that trades instance diversity against wall clock.

    Parameters:
        name : The name of the budget, used for artifact filenames.
        n_train : The satisfiable training instances to generate.
        n_valid : The tail of the training split held out for validation,
                  drawn from the training distribution rather than from the
                  ``(n, alpha)`` grid, which no training run may look at.
        n_test : The instances of each ``(n, alpha)`` test set.
        batch_size : The instances per compiled batch.
        pool_batches : The batches compiled and held at once.
        epochs : The passes over the pool between two refreshes.
        pools : The number of pools a run trains on, i.e. the outer loop.
        rounds : The message-passing rounds trained at.
        supervised : The last rounds a loss is put on, ``None`` for all of
                     them.
        lr : The learning rate of Adam.
        grid_n, alphas : The ``(n, alpha)`` cells this budget builds test
                         sets for and reports on.
        seeds : The seeds to train.
    """
    name: str
    n_train: int
    n_valid: int
    n_test: int
    batch_size: int
    pool_batches: int
    epochs: int
    pools: int
    rounds: int = 32
    supervised: int = None
    lr: float = 1e-3
    grid_n: tuple[int, ...] = GRID_N
    alphas: tuple[float, ...] = ALPHAS
    seeds: tuple[int, ...] = SEEDS

    @property
    def instances_per_pool(self) -> int:
        """ The distinct instances one pool holds. """
        return self.batch_size * self.pool_batches

    @property
    def steps(self) -> int:
        """ The optimizer steps a whole run takes. """
        return self.pools * self.epochs * self.pool_batches

    @property
    def grid(self) -> tuple:
        """ The ``(n, alpha)`` cells this budget reports on. """
        return tuple((n, alpha) for n in self.grid_n for alpha in self.alphas)


#: A few-minute miniature, faithful in everything but the amounts, so it
#: exercises exactly the code paths of the large run.  Its grid is a corner
#: of :data:`GRID` rather than the whole of it, because a test set costs a
#: compilation per batch just as a training pool does.
QUICK = Budget(
    name="quick", n_train=1000, n_valid=64, n_test=64, batch_size=8,
    pool_batches=8, epochs=4, pools=2, rounds=16, lr=1e-3,
    grid_n=(10, 25), alphas=(3.0, 4.26))

#: The budget behind the recorded baseline.  ``pools * instances_per_pool``
#: instances are seen in total; the pool is what a compilation buys.
FULL = Budget(
    name="full", n_train=40000, n_valid=1000, n_test=1000, batch_size=64,
    pool_batches=64, epochs=16, pools=10, rounds=32, lr=1e-3)

#: The widths of the recorded baseline.
WIDTHS = {"factor": Widths(dim=32, state_dim=64, hidden=128)}

#: Test-time compute swept after training, in message-passing rounds.
SWEEP = (32, 48, 64, 96, 128, 192, 256)

#: The noise added to the learned initial state, in units of its own scale:
#: the initial belief is one learned vector shared by every literal, so
#: without a symmetry-breaking perturbation every literal of a symmetric
#: formula would carry the same state for ever.
INIT_NOISE = 0.1

#: The weight of the auxiliary literal-consistency term in the loss.
CONSISTENCY = 0.1

#: The floor inside the logarithm of the smooth-SAT loss.
EPS = 1e-6
