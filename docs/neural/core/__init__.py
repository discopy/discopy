# -*- coding: utf-8 -*-

"""
What a *study* needs on top of a model: the benchmark kit.

The models' machinery is not here.  The wiring, the cells, the
interpretation and the message-passing schedules live in
:mod:`discopy.neural`, where they are generic in the task and in the
source category; what this package holds is everything a *benchmark*
of that machinery needs and no benchmark owns -- so that a new task is
exactly its four artefacts (combinatorics, signatures, encoder, decoder)
plus a configuration of values, and nothing else.

* :mod:`core.study` : the torch-free dataclasses a study is made of --
  :class:`~core.study.Widths`, :class:`~core.study.Budget`,
  :class:`~core.study.Split`.
* :mod:`core.train` : the epoch harness -- deep supervision, evaluation,
  batching -- generic in the task.
* :mod:`core.heads` : the fill-in-the-blanks encoder and decoder, shared
  by any benchmark whose input is a partial solution.
* :mod:`core.solvers` : the three solver shapes as templates -- the parts
  of each model, in the RNG-load-bearing order, written once.
* :mod:`core.registry` : the study protocol -- checkpointing, the cached
  training entry point, the learning-rate grid -- over a
  :class:`~core.registry.TaskSpec`.
* :mod:`core.act` : the adaptive-computation-time evaluations, bound to
  the family's decode rule.
* :mod:`core.recipes` : the training ingredients of the searched recipes
  -- optimizer, schedule, weight averaging and the segmented outer loop
  as one reusable function.
"""

__all__ = ["act", "heads", "recipes", "registry", "solvers", "study",
           "train"]
