# -*- coding: utf-8 -*-

"""
The shared machinery behind ``train_a_goi.py``, ``train_b_rrn.py`` and
``train_c_trm.py``: three sudoku solvers built as combinatorial maps in
:mod:`discopy.neural`.

* :mod:`experiments.config` : paths, seeds, budgets and the matched widths.
* :mod:`experiments.data` : the Palm et al. (2018) sudoku benchmark and the
  sudoku symmetry group.
* :mod:`experiments.skeleton` : the abstract, torch-free wirings of the two
  topologies -- the bipartite factor graph shared by models A and C, and the
  peer clique of model B -- typed by port *role* rather than width.
* :mod:`experiments.maps` : the neural functors interpreting each skeleton
  into concrete networks and widths, and the ``route`` helper that makes
  message passing resumable.
* :mod:`experiments.models` : the three solvers (GoI, RRN, TRM).
* :mod:`experiments.train` : training loops, evaluation and diagnostics.

Each training script imports the model and the data from here but writes out
its own epoch loop, since the loop is precisely where the three differ.
"""

__all__ = ["config", "data", "maps", "models", "skeleton", "train"]
