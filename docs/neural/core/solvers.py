# -*- coding: utf-8 -*-

"""
The three solver shapes of the study, as templates over any task.

A model of this family is a skeleton, an assignment of widths to roles, a
handful of modules built in a fixed order and a schedule -- and only the
first two are a task's.  What this module owns is the rest: which parts a
solver of each shape has, *in which order* they are built, and which
schedule runs them.  The order is load-bearing -- every constructor draws
from the global generator, so permuting the parts permutes every weight
under a seed -- which is exactly why it should be written once, here,
rather than once per benchmark.

* :func:`single_run` -- one differentiated run, a loss on every round:
  embedding, cell, optional relation, readout.  Model A is this with a
  factor graph, model B with a pairwise clique and no relation box.
* :func:`recursion` -- the segmented recursion of
  :cite:t:`JolicoeurMartineau25`, with or without a halt head: the same
  parts plus the answer refresh, its normalisation, and the learned
  initial answer, in that order, the halt head built last so that an ACT
  model under a seed is bitwise the plain recursion plus two constant
  tensors.

The task supplies the cell factories -- they read the task's roles and
mode mappings -- and the ``(box name, role)`` keys the engine reads;
everything a factory needs beyond that is a width or a class count.
"""

from __future__ import annotations

import torch

from discopy.neural.engine import ACTEngine, Engine, RecursionEngine
from discopy.neural.engine import Schedule
from core.heads import Decoder, Encoder

#: The single-run solvers supervise every round of one differentiated run.
DEEP = dict(cycles=1, steps=1, inject=True, supervise="round")

#: The recursion supervises every detached segment and carries its clues.
SEGMENTED = dict(inject=False, supervise="step")


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def single_run(skeleton, ob, cell, n_classes: int, dim: int, state_dim: int,
               rounds: int, inputs: tuple, state: tuple, relation=None,
               sites=None) -> Engine:
    """
    A single-run solver: deep supervision on every round of one backward
    graph.

    Parameters:
        skeleton : The abstract wiring to interpret.
        ob : The :class:`~discopy.neural.Dim` each atomic role carries.
        cell : A zero-argument factory of the shared cell module.
        n_classes : The number of classes the readout emits.
        dim : The width of the input embedding.
        state_dim : The width the readout decodes.
        rounds : The message-passing rounds of the run.
        inputs : The ``(box name, role)`` of the input trace.
        state : The ``(box name, role)`` the readout decodes.
        relation : A zero-argument factory of the shared relation module,
                   or ``None`` on a pairwise skeleton.
        sites : The module attribute filling each box name;
                ``{"cell": "cell", "unit": "factor"}`` or its pairwise
                restriction by default.
    """
    parts = {"embedding": lambda: Encoder(n_classes, dim), "cell": cell}
    if relation is not None:
        parts["factor"] = relation
    parts["readout"] = lambda: Decoder(state_dim, n_classes)
    return Engine(
        skeleton, ob, parts=parts,
        sites=sites or ({"cell": "cell", "unit": "factor"}
                        if relation is not None else {"cell": "cell"}),
        schedule=Schedule(rounds=rounds, **DEEP),
        inputs=inputs, state=state, n_classes=n_classes)


def recursion(skeleton, ob, cell, relation, n_classes: int, dim: int,
              state_dim: int, y_dim: int, rounds: int, cycles: int,
              n_sup: int, inputs: tuple, state: tuple, answer: tuple,
              sites=None, **kwargs):
    """
    A segmented-recursion solver, with a halt head when one is given.

    Parameters:
        skeleton : The abstract wiring to interpret.
        ob : The :class:`~discopy.neural.Dim` each atomic role carries.
        cell : A zero-argument factory of the shared cell module.
        relation : A zero-argument factory of the shared relation module.
        n_classes : The number of classes the readout emits.
        dim : The width of the input embedding.
        state_dim : The width of the latent state the refresh reads.
        y_dim : The width of the answer embedding.
        rounds : The rounds per cycle.
        cycles : The cycles per supervision step.
        n_sup : The supervision steps.
        inputs : The ``(box name, role)`` of the input trace.
        state : The ``(box name, role)`` of the latent state.
        answer : The ``(box name, role)`` of the answer trace.
        sites : The module attribute filling each box name.
        kwargs : Passed to the engine; a ``halt`` factory selects
                 :class:`~discopy.neural.engine.ACTEngine`.
    """
    parts = {
        "embedding": lambda: Encoder(n_classes, dim),
        "cell": cell,
        "factor": relation,
        "answer": lambda: torch.nn.GRUCell(state_dim, y_dim),
        "answer_norm": lambda: torch.nn.LayerNorm(y_dim),
        "readout": lambda: Decoder(y_dim, n_classes),
        "y0": lambda: torch.nn.Parameter(torch.zeros(y_dim))}
    factory = ACTEngine if "halt" in kwargs else RecursionEngine
    return factory(
        skeleton, ob, parts=parts,
        sites=sites or {"cell": "cell", "unit": "factor"},
        schedule=Schedule(rounds=rounds, cycles=cycles, steps=n_sup,
                          **SEGMENTED),
        inputs=inputs, state=state, n_classes=n_classes, answer=answer,
        **kwargs)
