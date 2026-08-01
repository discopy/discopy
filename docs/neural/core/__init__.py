# -*- coding: utf-8 -*-

"""
What a *study* needs on top of a model: the harness that trains and scores
one, and the dataclasses a run is configured by.

The models themselves are no longer here.  The wiring, the cells, the
interpretation and the message-passing schedules moved into
:mod:`discopy.neural`, where they are generic in the task and in the source
category; what is left is the part that is a study rather than a
formalism -- optimizers, epochs, evaluation protocols -- and that
deliberately stays outside the library.

* :mod:`core.study` : the torch-free dataclasses a study is made of --
  :class:`~core.study.Widths`, :class:`~core.study.Budget`,
  :class:`~core.study.Split`.
* :mod:`core.train` : the training harness -- deep supervision,
  evaluation, batching -- generic in the task.
* :mod:`core.recipes` : the training ingredients of the searched recipes
  -- optimizer, schedule, weight averaging and the segmented outer loop
  as one reusable function.
"""

__all__ = ["recipes", "study", "train"]
