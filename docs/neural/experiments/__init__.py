# -*- coding: utf-8 -*-

"""
The shared machinery behind ``train_a_goi.py``, ``train_b_rrn.py`` and
``train_c_trm.py``: three sudoku solvers built as combinatorial maps in
:mod:`discopy.neural`.

* :mod:`experiments.config` : paths, seeds, budgets and the matched widths.
* :mod:`experiments.data` : the Palm et al. (2018) sudoku benchmark and the
  sudoku symmetry group.
* :mod:`experiments.maps` : the wirings of the three maps and the ``route``
  helper that makes message passing resumable.
* :mod:`experiments.models` : the three solvers (GoI, RRN, TRM).
* :mod:`experiments.train` : training loops, evaluation and diagnostics.

Each training script imports the model and the data from here but writes out
its own epoch loop, since the loop is precisely where the three differ.
"""

__all__ = ["config", "data", "maps", "models", "train"]
