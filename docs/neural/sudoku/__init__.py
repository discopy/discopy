# -*- coding: utf-8 -*-

"""
The sudoku task: the engine of :mod:`discopy.neural` instantiated on the
game.

The package brings only what is irreducibly sudoku -- the grid
combinatorics, the roles its wires carry, an encoder, a decoder, the two
benchmarks and the recorded configurations of the study.  There is no cell
class and no solver class here: those are :mod:`discopy.neural`'s, and a
model is a choice of skeleton, widths and schedule.

* :mod:`sudoku.config` : grid constants, paths, seeds, budgets and the
  matched widths.
* :mod:`sudoku.signature` : the roles a port can play and the signature of
  each box -- what the wires mean, before they have a width.
* :mod:`sudoku.skeleton` : rows, columns, blocks and peers, and the two
  skeletons they induce via :mod:`discopy.neural.skeleton`.
* :mod:`sudoku.heads` : the historical import path of the encoder and
  decoder, which turned out to be the family's and live in
  :mod:`core.heads`.
* :mod:`sudoku.data` : the Palm et al. (2018) benchmark and the sudoku
  symmetry group.
* :mod:`sudoku.sudoku_extreme` : the sudoku-extreme benchmark with three
  pre-augmented training variants, loadable in place of the above.
* :mod:`sudoku.models` : models A, B and C as instantiations of the
  :mod:`core.solvers` templates -- which skeleton, at which widths, with
  which cell hyperparameters.
* :mod:`sudoku.act` : model C with a halt head, an import path over
  :mod:`core.act` and :func:`sudoku.models.act`.
* :mod:`sudoku.train` : the task's :class:`~core.registry.TaskSpec` and
  the historical entry points over :mod:`core.registry`; the baseline
  scripts ``train_a_goi.py``, ``train_b_rrn.py`` and ``train_c_trm.py``
  drive it, and the searched recipes live under ``best/``.
"""

__all__ = ["act", "config", "data", "heads", "models", "signature",
           "skeleton", "sudoku_extreme", "train"]
