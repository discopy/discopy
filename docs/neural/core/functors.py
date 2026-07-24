# -*- coding: utf-8 -*-

"""
The semantics of the solver family: functors filling a skeleton in.

The *syntax* of a solver -- which variable talks to which unit or peer --
is an abstract, torch-free skeleton built by :mod:`core.skeletons` from a
task's combinatorics, with atomic types naming roles rather than widths.
This module fills the skeleton in: a :class:`discopy.neural.Functor` sends
each role to the :class:`Dim` it carries and each abstract box to the
:class:`Network` computing it, and :func:`core.cmaps.interpret` applies
the functor, giving the closed :class:`discopy.neural.CMap` the solver
runs.  Two solvers may interpret the *same* skeleton with two different
functors: sending the ``answer`` role to ``Dim(0)``, the monoidal unit,
erases the answer loop altogether, sending it to ``Dim(y_dim)`` keeps it.

:func:`build` returns the interpreted map together with a :class:`Layout`,
a description of the port layout of a cell read off the same functor, so
that a solver knows where to inject clues and where to read states.
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy import frobenius
from discopy.neural import Dim, Functor, Network

from core import skeletons
from core.cmaps import interpret, roles_of


@dataclass(frozen=True)
class Layout:
    """
    Where a cell keeps what, as positions in its logical port order.

    Parameters:
        n_message : The number of message ports of a cell.
        message_width : The width of one message port.
        state : The pair of state-loop positions.
        clue : The pair of clue-loop positions.
        answer : The pair of answer-loop positions, possibly empty.
        state_width : The width of a state port.
        clue_width : The width of a clue port.
        answer_width : The width of an answer port.
        n_cells : The number of cell boxes, which come first in the map.
    """
    n_message: int
    message_width: int
    state: tuple[int, int]
    clue: tuple[int, int]
    answer: tuple[int, ...]
    state_width: int
    clue_width: int
    answer_width: int
    n_cells: int

    @property
    def state_offset(self) -> int:
        """ Where the state starts inside a cell's flat message vector. """
        return self.n_message * self.message_width


def layout_of(functor: Functor, abstract: frobenius.CMap) -> Layout:
    """
    The port layout of a cell under a functor: where each surviving role
    sits among the concrete ports of a cell box, and how wide it is.  Read
    off the same functor as :func:`core.cmaps.interpret`, so the two
    cannot drift.

    Parameters:
        functor : The neural functor giving the widths.
        abstract : The skeleton whose cells to lay out.
    """
    roles = roles_of(abstract, 0)
    kept = [role for role in roles if len(functor(role))]

    def positions(role):
        return tuple(i for i, other in enumerate(kept) if other == role)

    def width(role):
        return sum(functor(role).inside)

    message = kept[0]
    assert message not in skeletons.LOOP_ROLES, "a cell starts with a loop"
    return Layout(
        n_message=len(positions(message)), message_width=width(message),
        state=positions(skeletons.STATE), clue=positions(skeletons.CLUE),
        answer=positions(skeletons.ANSWER),
        state_width=width(skeletons.STATE),
        clue_width=width(skeletons.CLUE),
        answer_width=width(skeletons.ANSWER)
        if positions(skeletons.ANSWER) else 0,
        n_cells=sum(box.name == "cell" for box in abstract.boxes))


def role_functor(ob: dict, modules: dict,
                 abstract: frobenius.CMap) -> Functor:
    """
    The neural functor with the given object map, sending each abstract
    box of a skeleton to a :class:`Network` of the image type around the
    module of the same name.  One shared module means one shared box.

    Parameters:
        ob : Map from atomic role to the :class:`Dim` it carries.
        modules : Map from box name to the torch module filling it.
        abstract : The skeleton whose boxes to interpret.
    """
    types = Functor(ob=ob, dom=frobenius.Diagram)
    networks = {
        box: Network(box.name, types(box.dom), types(box.cod),
                     module=modules[box.name])
        for box in dict.fromkeys(abstract.boxes)}
    return Functor(ob=ob, ar=networks, dom=frobenius.Diagram)


def factor_functor(abstract: frobenius.CMap, cell_module, factor_module,
                   dim: int, state_dim: int,
                   answer_dim: int = 0) -> Functor:
    """
    The functor interpreting a :func:`core.skeletons.factor_graph`: the
    answer loop is erased when ``answer_dim`` is ``0`` and kept otherwise.

    Parameters:
        abstract : The factor-graph skeleton to interpret.
        cell_module : The shared module of every cell box.
        factor_module : The shared module of every unit box.
        dim : The width of a message and of a clue embedding.
        state_dim : The width of a variable's recurrent state.
        answer_dim : The width of the answer loop, ``0`` to erase it.
    """
    return role_functor(
        ob={skeletons.MESSAGE: Dim(dim), skeletons.STATE: Dim(state_dim),
            skeletons.CLUE: Dim(dim), skeletons.ANSWER: Dim(answer_dim)},
        modules={"cell": cell_module, "unit": factor_module},
        abstract=abstract)


def clique_functor(abstract: frobenius.CMap, cell_module, state_dim: int,
                   dim: int) -> Functor:
    """
    The functor interpreting a :func:`core.skeletons.clique`: a peer wire
    carries a full hidden state of width ``state_dim``, and the state loop
    carries the concatenated hidden and cell state of an ``LSTMCell``,
    hence its width of ``2 * state_dim``.

    Parameters:
        abstract : The clique skeleton to interpret.
        cell_module : The shared module of every cell box.
        state_dim : The width of a hidden state, i.e. of a peer wire.
        dim : The width of a clue embedding.
    """
    return role_functor(
        ob={skeletons.PEER: Dim(state_dim),
            skeletons.STATE: Dim(2 * state_dim),
            skeletons.CLUE: Dim(dim)},
        modules={"cell": cell_module},
        abstract=abstract)


def build(functor: Functor, abstract: frobenius.CMap) -> tuple:
    """
    The interpreted map and its layout, from one functor: what a solver
    needs to run.

    Parameters:
        functor : The neural functor giving the widths and the networks.
        abstract : The closed skeleton to interpret.
    """
    return interpret(functor, abstract), layout_of(functor, abstract)
