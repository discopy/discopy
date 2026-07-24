# -*- coding: utf-8 -*-

"""
The general library between :mod:`discopy.neural` and a task package:
the solver family and everything needed to build, train and evaluate it.

A task (see ``sudoku/`` for the worked example) brings only its
combinatorics, its datasets and its recorded configurations; this package
holds the method:

* :mod:`core.study` : the torch-free dataclasses a study is made of --
  :class:`~core.study.Widths`, :class:`~core.study.Budget`,
  :class:`~core.study.Split`.
* :mod:`core.cmaps` : reading a closed map back into boxes, wires and
  roles, :func:`~core.cmaps.interpret` turning a skeleton and a functor
  into a runnable :class:`discopy.neural.CMap`, and the
  :class:`~core.cmaps.Router` that makes message passing resumable.
* :mod:`core.skeletons` : the port roles and the two abstract wirings of
  the family -- the bipartite factor graph and the pairwise clique --
  parameterized by a task's membership structure.
* :mod:`core.functors` : the functors filling a skeleton in, and the
  :class:`~core.functors.Layout` a solver reads its ports by.
* :mod:`core.solvers` : the update cells and the three solver
  architectures -- factor graph, clique, segmented recursion.
* :mod:`core.act` : adaptive computation time for the recursion solver --
  the halt head, the slot-refill trainer and early-stopping inference.
* :mod:`core.train` : the training harness -- deep supervision,
  evaluation, batching -- generic in the task.
* :mod:`core.recipes` : the training ingredients of the searched recipes
  -- optimizer, schedule, weight averaging and the segmented outer loop
  as one reusable function.
"""

__all__ = ["act", "cmaps", "functors", "recipes", "skeletons", "solvers",
           "study", "train"]
