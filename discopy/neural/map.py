# -*- coding: utf-8 -*-

"""
From a diagram to a global interaction: what a generator means, and what a
whole diagram compiles to.

A generator :math:`f : X \\to Y` of the source category is interpreted on
the *boundary* of its box,

.. math:: \\partial f = X^* \\otimes Y,

the inputs read as the outputs of whatever is upstream, together with the
outputs: its module is a local interaction

.. math:: \\Phi_f : \\partial f \\otimes P_f \\to \\partial f,

reading one incoming message and emitting one outgoing message on every
port, the local half of the execution formula of the geometry of
interaction :cite:p:`Abramsky96`.  Why that is not an ordinary parametric
map :math:`X \\otimes P \\to Y`, and why two of them do not compose by
substitution, is said once in :class:`~discopy.neural.Network`.  This
module is the functor that sends a whole diagram to the closed
:class:`~discopy.neural.CMap` running every local interaction at once, and
the addressing of that map's state by ``(generator name, role)``.

The global interaction
----------------------

A diagram wires the boundaries together.  Interpreting it -- each atomic
role to the :class:`~discopy.neural.Dim` it carries, each generator name to
the module computing it -- gives a closed :class:`~discopy.neural.CMap`,
and one synchronous round of message passing is the two halves in
sequence: every box interacts with the messages on its own ports, then the
wires carry each emission to the other end.  Writing :math:`\\Phi_\\theta`
for the parallel application of every local interaction and
:math:`\\sigma_D` for the permutation of the flat state induced by the
wiring,

.. math:: T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S_D \\to S_D

on the state object :math:`S_D = \\bigoplus_p \\mathbb{R}^{w_p}`, one summand
per port.  :func:`interpret` builds that closed map, :func:`families` the
port index of every ``(generator name, role)`` family and :func:`heads` the
ports of a family a module reads a value off, and
:meth:`~discopy.neural.CMap.forward` with ``return_flat`` is the one
implementation of :math:`T^n`.

When an initial message vector :math:`i` is re-injected -- ``inject=True``
-- the round is

.. math:: T_{D,\\theta,i}(s) = \\sigma_D(\\Phi_\\theta(s)) + i,

an affine, not a linear, dependence on :math:`i`: the vector is added back
to the *whole* state after routing, every round.

Four notions that are easy to conflate, kept apart
--------------------------------------------------

* a **categorical trace** is a structural operation on wiring.  A
  self-wired pair of ports *is* the trace of the compact target, and a
  functor into :mod:`discopy.neural` preserves it strictly and for free.
* a **persistent state channel** -- delayed feedback in the sense of
  :mod:`discopy.feedback` -- is what that same pair does across rounds:
  what a box writes on one end it reads on the other one round later.
* **finite iteration** is what running the map computes.  ``n`` rounds
  compute :math:`T^n(s_0)` and nothing more.  What holds unconditionally is
  resumption, :math:`T^{a+b} = T^b \\circ T^a`, which is why a segmented
  solver can stop and carry on -- and it holds for *one* transition, so a
  run resumed from its own carried state only resumes when ``inject`` is
  off.
* a **fixed point** of :math:`T` is a fourth thing.  If some :math:`T`
  happens to be a contraction then :math:`T^n` converges, but that is an
  analytic property of the learned weights, to be measured -- never
  something the category supplies: the residual :math:`\\|T(s) - s\\|` of
  a state is the number that says whether it is one.

Note
----
``X*`` is represented by the same :class:`~discopy.neural.Dim` data as
``X``, because every atomic dimension is self-dual.  The *order* is where
the two differ: ``Dim(2, 3).r == Dim(3, 2)`` reverses a composite type,
whereas a module reads its domain ports in domain order -- which is what
:meth:`~discopy.cmap.CMap.box_ports` restores when it un-reverses the
clockwise storage.  So the boundary a module reads is ``dom @ cod``.

This module imports no tensor framework, so that a diagram can be compiled
and inspected on a machine without one.

Summary
-------

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        to_map
        functor
        interpret
        heads
        families
        width
"""

from __future__ import annotations

from typing import Mapping

from discopy import cmap
from discopy.neural.core import CMap, Functor, Network
from discopy.utils import assert_isinstance


def to_map(source):
    """ The map of a closed diagram, or the map itself. """
    return source if isinstance(source, cmap.CMap) else source.to_map()


def functor(source, ob: Mapping, ar: Mapping = None) -> Functor:
    """
    The neural functor :math:`F_\\theta` of an interpretation: each atomic
    role to the :class:`~discopy.neural.Dim` it carries, each generator to
    the :class:`~discopy.neural.Network` of the image type around the
    module of the same name.

    Without ``ar`` it is the functor on objects alone, enough to read the
    width every role carries: an integer is read as an atomic dimension
    and a dualised role as the dual of its image, so ``{x: 3}`` sends
    ``x.r`` to ``Dim(3)`` as it sends ``x``.

    Parameters:
        source : The closed diagram or map in the source category.
        ob : The ``Dim`` each atomic role carries, ``Dim(0)`` to erase it.
        ar : The module filling each generator name.  One shared module
             means one shared box, hence one batched call per round for a
             whole family of sites.

    Example
    -------
    >>> from discopy.compact import Box, Ty
    >>> x = Ty("x")
    >>> image = functor(Box("f", Ty(), x @ x.r).to_map(), {x: 3})
    >>> image(x), image(x.r), image(Ty())
    (Dim(3), Dim(3), Dim(0))
    """
    category = type(to_map(source)).category
    image = Functor(ob_map=dict(ob), dom=category)
    if ar is None:
        return image
    networks = {
        box: Network(box.name, image(box.dom), image(box.cod),
                     module=ar[box.name])
        for box in dict.fromkeys(source.boxes)}
    return Functor(ob_map=dict(ob), ar_map=networks, dom=category)


def interpret(source, ob: Mapping, ar: Mapping) -> CMap:
    """
    Compile a closed diagram into the :class:`~discopy.neural.CMap` that
    runs it, port by port: each generator becomes its image
    :class:`~discopy.neural.Network`, each wire a wire between the image
    ports. The ``(generator name, role)`` addressing of its flat state is
    :func:`families`.

    A role must go to an atomic ``Dim`` -- one abstract port becomes one
    concrete port -- or to ``Dim(0)``, in which case the port vanishes and
    the wire on it with it.  A wire joins two ports of adjoint roles, which
    the functor sends to dimensions of one width, so a wire is erased whole
    or not at all.

    Parameters:
        source : The closed map in the source category, whose atomic types
                 name the *role* a port plays rather than its width; a
                 diagram is read through
                 :meth:`~discopy.cmap.CMap.from_diagram`.
        ob : The ``Dim`` each atomic role carries.
        ar : The module filling each generator name.

    Example
    -------
    >>> from discopy.frobenius import Box, Diagram, Ty
    >>> from discopy.neural import Dim
    >>> x = Ty("x")
    >>> f, g = Box("f", Ty(), x @ x), Box("g", x @ x, Ty())
    >>> compiled = interpret(f >> g, {x: Dim(2)}, {"f": None, "g": None})
    >>> tuple(compiled.edges), Dim(*compiled.port_widths)
    ((3, 2, 1, 0), Dim(2, 2, 2, 2))
    >>> tuple(interpret(f >> Diagram.swap(x, x) >> g,
    ...                 {x: Dim(2)}, {"f": None, "g": None}).edges)
    (2, 3, 0, 1)
    """
    source = to_map(source)
    if len(source.dom) or len(source.cod):
        raise ValueError("only a closed diagram compiles to a map")
    image = functor(source, ob, ar)
    boxes = tuple(image(box) for box in source.boxes)
    for network in boxes:
        assert_isinstance(network, Network)

    position = {}
    for index, box in enumerate(source.boxes):
        cursor = 0
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            width = image(role)
            if len(width) > 1:
                raise ValueError(f"{role} maps to the non-atomic {width}")
            if len(width):
                position[index, place] = cursor
            cursor += len(width)

    logical = {port: (index, place)
               for index in range(len(source.boxes))
               for place, port in enumerate(source.box_ports(index))}
    wires = [
        ((logical[port][0], position[logical[port]]),
         (logical[other][0], position[logical[other]]))
        for port, other in enumerate(source.edges)
        if port < other and logical[port] in position]
    return CMap.from_wiring(boxes, wires)


def heads(source) -> dict:
    """
    The *heads* of each ``(generator name, role)`` family of a closed
    diagram, as ``(box index, port position)`` pairs in box order then
    position order: the ports a module reads a value off rather than the
    far end of its own loop. A port is a head unless it is wired to an
    earlier port of the same box, which is exactly the second copy of a
    traced leg, read off the wiring rather than off a declaration.

    Parameters:
        source : The closed diagram or map in the source category.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> heads(pair)["cell", state]
    ((0, 1), (1, 1))
    """
    source = to_map(source)
    result: dict = {}
    for index, box in enumerate(source.boxes):
        ports = source.box_ports(index)
        place_of = {port: place for place, port in enumerate(ports)}
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            if place_of.get(source.edges[ports[place]], place) >= place:
                result.setdefault((box.name, role), []).append((index, place))
    return {key: tuple(value) for key, value in result.items()}


def families(source, cmap: CMap, ob: Mapping) -> tuple[dict, dict]:
    """
    The global port indices of each ``(generator name, role)`` pair of a
    compiled diagram, in box order then position order: every port of the
    family, and its :func:`heads`.

    Parameters:
        source : The closed map that was compiled.
        cmap : Its image under :func:`interpret`.
        ob : The ``Dim`` each atomic role carries, as given to
             :func:`interpret`.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> ob = {peer: Dim(3), state: Dim(5)}
    >>> kept = interpret(pair, ob, {"cell": None})

    The ports of a map are stored clockwise, so the codomain of each box
    comes last and reversed:

    >>> kept.port_widths
    (5, 5, 3, 5, 5, 3)
    >>> ports, heads = families(pair, kept, ob)
    >>> ports["cell", state], heads["cell", state]
    ((1, 0, 4, 3), (1, 4))

    Sending a role to ``Dim(0)`` erases its ports and the wires on them,
    which is how one diagram serves two models:

    >>> ob = {peer: Dim(3), state: Dim(0)}
    >>> erased = interpret(pair, ob, {"cell": None})
    >>> erased.port_widths, ("cell", state) in families(pair, erased, ob)[0]
    ((3, 3), False)
    """
    source = to_map(source)
    image = functor(source, ob)
    is_head = {place for places in heads(source).values() for place in places}
    ports: dict = {}
    head_ports: dict = {}
    for index, box in enumerate(source.boxes):
        concrete, cursor = cmap.box_ports(index), 0
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            if not len(image(role)):
                continue
            port = concrete[cursor]
            cursor += 1
            ports.setdefault((box.name, role), []).append(port)
            if (index, place) in is_head:
                head_ports.setdefault((box.name, role), []).append(port)
    return ({key: tuple(value) for key, value in ports.items()},
            {key: tuple(value) for key, value in head_ports.items()})


def width(source, ob: Mapping) -> int:
    """
    The flat state width of a closed diagram under an interpretation, i.e.
    the sum over its ports of the dimension each role carries, read off the
    diagram without compiling it.

    Parameters:
        source : The closed diagram or map in the source category.
        ob : The width each atomic role carries, as an integer or a
             :class:`~discopy.neural.Dim`.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> width(pair, {peer: 3, state: 5})
    26
    >>> width(pair, {peer: 3, state: 5}) == sum(
    ...     interpret(pair, {peer: Dim(3), state: Dim(5)}, {"cell": None}
    ...               ).port_widths)
    True
    """
    source = to_map(source)
    image = functor(source, ob)
    return sum(sum(image(role).inside) for box in source.boxes
               for role in tuple(box.dom) + tuple(box.cod))
