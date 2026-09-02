# -*- coding: utf-8 -*-

"""
From a diagram to a global interaction: what a generator means, and what a
whole diagram compiles to.

Two notions, and the difference between them is the whole point of this
package.

An **ordinary parametric map** :math:`(P, f) : X \\to Y` is a map

.. math:: f : X \\otimes P \\to Y

from an input and a parameter object to an output, the parameters on the
right as :mod:`discopy.para` has them.  That is a feed-forward
layer, and it is *not* what a :class:`~discopy.neural.Network` is.  These
are the morphisms of :math:`\\mathrm{Para}`, they compose by substitution,
and :class:`ParamMap` records them.

A **parametric interaction map** :math:`(P, \\Phi) : X \\to Y` is a map on
the *boundary* of a box.  Write the boundary as

.. math:: \\partial f = X^* \\otimes Y,

the inputs read as the outputs of whatever is upstream, together with the
outputs; then a local neural interaction is

.. math:: \\Phi_f : \\partial f \\otimes P_f \\to \\partial f,

reading one incoming message on every port and emitting one outgoing
message on every port.  This is the local half of the execution formula of
the geometry of interaction :cite:p:`Abramsky96`, and it is exactly what
the torch module of a :class:`~discopy.neural.Network` computes:
``R ** width -> R ** width`` for ``width`` the sum of the domain and
codomain dimensions.  A module answers *every* leg of its box, its inputs
included, which no ``X -> Y`` signature can say.  :class:`InteractionMap`
records these, and deliberately does **not** compose: two interaction maps
glued along a shared object do not compose by substitution, they talk to
each other along wires.  That is the next paragraph.

The global interaction
----------------------

A diagram wires the boundaries together.  Interpreting it -- each atomic
role to the :class:`~discopy.neural.Dim` it carries, each generator name to
the torch module computing it -- gives a closed
:class:`~discopy.neural.CMap`, and one synchronous round of message passing
is the two halves in sequence: every box interacts with the messages on its
own ports, then the wires carry each emission to the other end.  Writing
:math:`\\Phi_\\theta` for the parallel application of every local
interaction and :math:`\\sigma_D` for the permutation of the flat state
induced by the wiring,

.. math:: T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta : S_D \\to S_D

on the state object :math:`S_D = \\bigoplus_p \\mathbb{R}^{w_p}`, one summand
per port.  :func:`interpret` builds that closed map, :func:`families` the
port index of every ``(generator name, role)`` family, and
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
:func:`box_ports` restores when it un-reverses the clockwise storage.  So
:attr:`InteractionMap.boundary` is ``dom @ cod``.

This module imports no tensor framework, so that a diagram can be compiled
and inspected on a machine without one.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    ParamMap
    InteractionMap

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        interaction_spec
        functor
        interpret
        families
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from discopy import para
from discopy.neural.core import (
    CMap, Diagram, Dim, Functor, Network, box_ports)
from discopy.utils import AxiomError, assert_isinstance, unbiased


@dataclass
class ParamMap(para.Symmetric):
    """
    An ordinary parametric map :math:`(P, f) : X \\to Y`, i.e. a map
    :math:`f : X \\otimes P \\to Y`: a parametric map of :mod:`discopy.para`
    over neural diagrams, whose ``inside`` is the box of the generator.

    A layer of a feed-forward network is one of these, and they compose by
    substitution: :math:`(Q, g) \\circ (P, f)` has parameter object
    :math:`P \\otimes Q`, the composition of parametric maps.  A
    :class:`~discopy.neural.Network` is *not* one: see
    :class:`InteractionMap`.

    Parameters:
        dom : The domain :math:`X`.
        cod : The codomain :math:`Y`.
        inside : The diagram ``dom @ param -> cod``.
        param : The parameter object :math:`P`, the unit by default.
        copar : Unused, the unit.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> f = ParamMap.generator("f", Dim(2), Dim(3), Dim(6))
    >>> g = ParamMap.generator("g", Dim(3), Dim(4), Dim(12))
    >>> (f >> g).dom, (f >> g).cod, (f >> g).params
    (Dim(2), Dim(4), Dim(6, 12))
    >>> (f @ g).dom, (f @ g).cod, (f @ g).params
    (Dim(2, 3), Dim(3, 4), Dim(6, 12))
    >>> (f >> g).inside.boxes[1]
    neural.core.Network('g', Dim(3, 12), Dim(4))
    >>> data = lambda one: (one.dom, one.cod, one.params)
    >>> data(ParamMap.id(Dim(2)) >> f) == data(f) == data(
    ...     f >> ParamMap.id(Dim(3)))
    True
    """
    category = Diagram

    @classmethod
    def generator(cls, name: str, dom: Dim, cod: Dim, params: Dim = Dim()
                  ) -> ParamMap:
        """
        The parametric map of a generator: its box ``dom @ params -> cod``
        with the parameters as parameter object.

        Parameters:
            name : The name of the generator.
            dom : The domain :math:`X`.
            cod : The codomain :math:`Y`.
            params : The parameter object :math:`P`, the unit by default.
        """
        return cls(dom, cod, Network(name, dom @ params, cod), params)

    @property
    def name(self) -> str:
        """ The name of the generator, or the diagram inside as a string. """
        return self.inside.name if isinstance(self.inside, Network) \
            else str(self.inside)

    @property
    def params(self) -> Dim:
        """ The parameter object :math:`P`, i.e. :attr:`param`. """
        return self.param

    @unbiased
    def then(self, other: ParamMap) -> ParamMap:
        assert_isinstance(other, ParamMap)
        return super().then(other)

    @unbiased
    def tensor(self, other: ParamMap) -> ParamMap:
        assert_isinstance(other, ParamMap)
        return super().tensor(other)


@dataclass
class InteractionMap(para.Symmetric):
    """
    A parametric interaction map :math:`(P, \\Phi) : X \\to Y`, i.e. a map
    :math:`\\Phi : (X^* \\otimes Y) \\otimes P \\to X^* \\otimes Y` on the
    boundary of a box: a parametric map of :mod:`discopy.para` whose domain
    and codomain are both the :attr:`boundary`, in the port order the
    executable module reads, the :attr:`inputs` then the :attr:`outputs`.

    This is the formal reading of a :class:`~discopy.neural.Network`: same
    name, same inputs, same outputs, and a module whose input and output
    both live on the boundary.

    Parameters:
        dom : The boundary :math:`X^* \\otimes Y`, as ``inputs @ outputs``.
        cod : The boundary again.
        inside : The box ``boundary @ param -> boundary``.
        param : The parameter object :math:`P`, the unit by default.
        copar : Unused, the unit.
        inputs : The inputs :math:`X` of the box.
        outputs : The outputs :math:`Y` of the box.

    Note
    ----
    Interaction maps do **not** compose by substitution, and ``>>`` refuses
    rather than pretending.  Two of them glued along a shared object talk
    to each other along the wires: that is symmetric feedback -- the trace
    of the two boxes over the shared boundary -- and what computes it is a
    finite number of rounds of :meth:`CMap.forward
    <discopy.neural.CMap.forward>`.  Their *tensor*
    is meaningful and is kept, because :math:`\\Phi_\\theta` is exactly the
    parallel application of every local interaction.

    Example
    -------
    >>> from discopy.neural import Dim
    >>> f = InteractionMap.generator("f", Dim(2), Dim(3), Dim(25))
    >>> g = InteractionMap.generator("g", Dim(5), Dim(7), Dim(49))
    >>> f.boundary, f.width
    (Dim(2, 3), 5)
    >>> f >> g
    Traceback (most recent call last):
        ...
    discopy.utils.AxiomError: interaction maps do not compose by \
substitution: wire them together and iterate, see CMap.forward.

    The boundary of a tensor is not the tensor of the boundaries -- it
    interleaves, which is why a map lays out one contiguous block per
    *box* rather than one per boundary:

    >>> (f @ g).boundary, f.boundary @ g.boundary
    (Dim(2, 5, 3, 7), Dim(2, 3, 5, 7))
    >>> (f @ g).width == f.width + g.width
    True
    """
    category = Diagram
    inputs: Dim = Dim()
    outputs: Dim = Dim()

    @classmethod
    def generator(cls, name: str, inputs: Dim, outputs: Dim,
                  params: Dim = Dim()) -> InteractionMap:
        """
        The interaction map of a generator: its box on the boundary
        ``inputs @ outputs``, with the parameters as parameter object.

        Parameters:
            name : The name of the generator.
            inputs : The inputs :math:`X` of the box.
            outputs : The outputs :math:`Y` of the box.
            params : The parameter object :math:`P`, the unit by default.
        """
        boundary = inputs @ outputs
        return cls(boundary, boundary,
                   Network(name, boundary @ params, boundary), params,
                   inputs=inputs, outputs=outputs)

    @property
    def name(self) -> str:
        """ The name of the generator. """
        return self.inside.name

    @property
    def params(self) -> Dim:
        """ The parameter object :math:`P`, i.e. :attr:`param`. """
        return self.param

    @property
    def boundary(self) -> Dim:
        """
        The boundary :math:`\\partial f = X^* \\otimes Y`, in the port order
        the executable module reads: the inputs then the outputs.
        """
        return self.dom

    @property
    def width(self) -> int:
        """ The flat width of the boundary, i.e. of the module. """
        return sum(self.boundary.inside)

    def dagger(self) -> InteractionMap:
        """
        The same interaction read backwards, :math:`(P, \\Phi)^\\dagger : Y
        \\to X`.

        Its boundary is the same object up to the symmetry exchanging the
        two halves, which is why
        :meth:`~discopy.neural.Network.dagger` can reuse the module: the
        weights are read in the new port order.

        Example
        -------
        >>> from discopy.neural import Dim
        >>> f = InteractionMap.generator("f", Dim(2), Dim(3), Dim(25))
        >>> f.dagger().boundary, f.dagger().width
        (Dim(3, 2), 5)
        >>> f.dagger().dagger() == f
        True
        """
        return self.generator(
            self.name, self.outputs, self.inputs, self.params)

    @unbiased
    def then(self, other) -> InteractionMap:
        raise AxiomError(
            "interaction maps do not compose by substitution: wire them "
            "together and iterate, see CMap.forward.")

    @unbiased
    def tensor(self, other: InteractionMap) -> InteractionMap:
        assert_isinstance(other, InteractionMap)
        return self.generator(
            f"({self.name} @ {other.name})", self.inputs @ other.inputs,
            self.outputs @ other.outputs, self.params @ other.params)


def interaction_spec(network) -> InteractionMap:
    """
    The formal interaction map a :class:`~discopy.neural.Network` realises.

    Read-only and side-effect free: it registers nothing, wraps nothing,
    owns no parameter and takes no part in a forward pass.  The network is
    the implementation; this is what it implements.

    The parameter object is ``Dim(n)`` for ``n`` the number of scalars the
    module holds, and the unit when the network has no module or its data
    is not a torch one -- reading it needs no ``torch`` import, only
    ``module.parameters()``.

    Parameters:
        network : The network to read.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.neural import Dim, Network
    >>> f = Network("f", Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    >>> spec = interaction_spec(f)
    >>> spec.name, spec.boundary, spec.params
    ('f', Dim(2, 3), Dim(30))
    >>> spec.width == f.module.in_features == f.module.out_features
    True
    >>> interaction_spec(Network("g", Dim(2), Dim(3))).params
    Dim(0)
    """
    module = getattr(network, "module", None)
    parameters = getattr(module, "parameters", None)
    params = Dim() if parameters is None else Dim(
        sum(parameter.numel() for parameter in parameters()))
    return InteractionMap.generator(
        network.name, network.dom, network.cod, params)


def functor(source, ob: Mapping, ar: Mapping) -> Functor:
    """
    The neural functor :math:`F_\\theta` of an interpretation: each atomic
    role to the :class:`~discopy.neural.Dim` it carries, each generator to
    the :class:`~discopy.neural.Network` of the image type around the
    module of the same name.

    Parameters:
        source : The closed map in the source category to interpret.
        ob : The ``Dim`` each atomic role carries, ``Dim(0)`` to erase it.
        ar : The torch module filling each generator name.  One shared
             module means one shared box, hence one batched call per round
             for a whole family of sites.
    """
    category = type(source).category.ar
    types = Functor(ob_map=dict(ob), dom=category)
    networks = {
        box: Network(box.name, types(box.dom), types(box.cod),
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
    the wire is erased with it.  A wire may only be erased whole: a role
    wired to a surviving role cannot be erased.

    Parameters:
        source : The closed map in the source category, whose atomic types
                 name the *role* a port plays rather than its width; a
                 diagram is read through
                 :meth:`~discopy.cmap.CMap.from_diagram`.
        ob : The ``Dim`` each atomic role carries.
        ar : The torch module filling each generator name.

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
    if not hasattr(source, "edges"):
        source = source.to_map()
    if len(source.dom) or len(source.cod):
        raise ValueError("only a closed diagram compiles to a map")
    image = functor(source, ob, ar)
    boxes = tuple(image(box) for box in source.boxes)
    for box, network in zip(source.boxes, boxes):
        assert_isinstance(network, Network)
        if (network.dom, network.cod) != (image(box.dom), image(box.cod)):
            raise ValueError(f"the image of {box} has the wrong type")

    erased, position = {}, {}
    for index, box in enumerate(source.boxes):
        cursor = 0
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            width = image(role)
            if len(width) > 1:
                raise ValueError(f"{role} maps to the non-atomic {width}")
            erased[index, place] = not len(width)
            position[index, place] = cursor
            cursor += len(width)

    logical = {port: (index, place)
               for index in range(len(source.boxes))
               for place, port in enumerate(box_ports(source, index))}
    wires = []
    for port, other in enumerate(source.edges):
        if port > other:
            continue
        one, two = logical[port], logical[other]
        if erased[one] and erased[two]:
            continue
        if erased[one] or erased[two]:
            raise ValueError(
                f"the wire {one} -- {two} is only erased at one end")
        wires.append(((one[0], position[one]), (two[0], position[two])))
    return CMap.from_wiring(boxes, wires)


def families(source, cmap: CMap, ob: Mapping) -> tuple[dict, dict]:
    """
    The global port indices of each ``(generator name, role)`` pair of a
    compiled diagram, in box order then position order: every port of the
    family, and its *heads*, the ports a module reads a value off rather
    than the far end of its own loop -- a port is a head unless it is wired
    to an earlier port of the same box, which is exactly the second copy
    of a traced leg, read off the wiring rather than off a declaration.

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
    >>> kept.port_widths  # ports are stored clockwise, codomain last
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
    if not hasattr(source, "edges"):
        source = source.to_map()
    ports: dict = {}
    heads: dict = {}
    for index, box in enumerate(source.boxes):
        abstract, concrete = box_ports(source, index), box_ports(cmap, index)
        place_of = {port: place for place, port in enumerate(abstract)}
        cursor = 0
        for place, role in enumerate(tuple(box.dom) + tuple(box.cod)):
            if not len(ob[role]):
                continue
            port = concrete[cursor]
            cursor += 1
            ports.setdefault((box.name, role), []).append(port)
            if place_of.get(source.edges[abstract[place]], place) >= place:
                heads.setdefault((box.name, role), []).append(port)
    return ({key: tuple(value) for key, value in ports.items()},
            {key: tuple(value) for key, value in heads.items()})
