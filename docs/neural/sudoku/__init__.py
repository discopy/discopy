# -*- coding: utf-8 -*-

"""
The sudoku task: the :mod:`core` solver family instantiated on the game.

The package brings only what is irreducibly sudoku -- the grid
combinatorics, the two benchmarks and the recorded configurations of the
study -- and binds the family's solvers to them:

* :mod:`sudoku.config` : grid constants, paths, seeds, budgets and the
  matched widths.
* :mod:`sudoku.skeleton` : rows, columns, blocks and peers, and the two
  skeletons they induce via :mod:`core.skeletons`.
* :mod:`sudoku.data` : the Palm et al. (2018) benchmark and the sudoku
  symmetry group.
* :mod:`sudoku.sudoku_extreme` : the sudoku-extreme benchmark with three
  pre-augmented training variants, loadable in place of the above.
* :mod:`sudoku.models` : models A, B and C -- the family's three
  architectures bound to the sudoku skeletons and widths.
* :mod:`sudoku.act` : model C with the halt head of :mod:`core.act`.
* :mod:`sudoku.train` : the study protocol -- checkpointing, the
  registry-bound entry point, the learning-rate grid -- on the harness of
  :mod:`core.train`; the baseline scripts ``train_a_goi.py``,
  ``train_b_rrn.py`` and ``train_c_trm.py`` drive it, and the searched
  recipes live under ``best/``.
"""

__all__ = ["act", "config", "data", "models", "skeleton",
           "sudoku_extreme", "train"]
