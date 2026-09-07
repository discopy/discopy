# -*- coding: utf-8 -*-

"""
The execution formula of the geometry of interaction, on any backend.

:class:`Execution` runs a neural :class:`~discopy.neural.CMap` on whichever
:class:`~discopy.neural.Backend` holds the tensors, torch or JAX. All the
messages live in one flat array of shape ``(batch_size, total width)``, in
port order; one round applies every box then routes along the wires. The
boxes sharing a module and a port signature are applied in one batched call
-- the geometry of interaction says nothing about the order the boxes fire
in, so the activation of a round is one call per *group* -- and the routing
is one permutation of the last axis. The private memory of a
:class:`~discopy.neural.Network` is a second flat array beside the messages,
and a feed-forward map can run its boxes once each, in topological order.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Execution

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        activate_box
        activate
        make_step
        box_forward
"""

from __future__ import annotations

from functools import cached_property

from discopy.cmap import PortKind
from discopy.neural.backend import Backend, get_backend


class Execution:
    """
    The geometry-of-interaction execution of a neural combinatorial map.

    Let ``edges`` be the fixpoint-free involution on ports and ``activate``
    apply every network at once. One synchronous round activates the boxes,
    then routes their outgoing messages with ``incoming = outgoing[src]``
    for ``src`` the flat index of ``edges``. Boundary inputs are emitted
    before every round, while ``init`` is optionally injected after every
    routing step as well as before the first round.

    Parameters:
        inside : The combinatorial map to execute.
        x : The boundary input, ``(batch_size, sum of domain widths)``.
        init : The initial incoming messages, per port or as one flat array.
        memory : The initial private memory, per box occurrence or flat.
        backend : The execution backend, the current backend by default.
        modules : The backend-owned modules, in the map's unique-module order.
                  The modules inside the boxes are used by default, so that
                  a backend can train copies of them without rebuilding
                  the map.

    Example
    -------
    >>> from discopy.neural import Dim, Network
    >>> f = Network('f', Dim(2), Dim(3), module=object())
    >>> execution = Execution(f.to_map(), backend="jax")  # doctest: +EXTRA
    >>> execution.inside.routing["src"]  # doctest: +EXTRA
    (2, 3, 0, 1, 7, 8, 9, 4, 5, 6)
    """
    def __init__(
            self, inside, x=None, init=None, memory=None,
            backend: str | Backend = None, modules=None):
        self.inside = inside
        self.x, self.init, self.memory = x, init, memory
        self.backend = get_backend(backend)
        self.modules = inside.modules if modules is None else tuple(modules)
        if len(self.modules) != len(inside.modules):
            raise ValueError(
                f"Expected {len(inside.modules)} modules, "
                f"got {len(self.modules)}.")
        self.batch_size, self.prototype = 1, None
        self.source = self.initial = self.incoming = self.outgoing = None
        self.stored = None

    @cached_property
    def topological_order(self) -> tuple[int, ...]:
        """ Order boxes from boundary inputs towards boundary outputs. """
        inside = self.inside
        if inside.loops:
            raise ValueError(
                "A causal schedule requires an acyclic map, without loops.")
        box_ports = tuple(
            inside.box_ports(i) for i in range(len(inside.boxes)))
        domain_owner, codomain_owner, box_port_owner = {}, {}, {}
        for box_index, (box, ports) in enumerate(zip(inside.boxes, box_ports)):
            arity = len(box.dom)
            for port in ports:
                box_port_owner[port] = box_index
            for port in ports[:arity]:
                domain_owner[port] = box_index
            for port in ports[arity:]:
                codomain_owner[port] = box_index

        dependencies = []
        for box_index, (box, ports) in enumerate(zip(inside.boxes, box_ports)):
            arity, current = len(box.dom), set()
            for port in ports[:arity]:
                source = inside.edges[port]
                if source in codomain_owner:
                    current.add(codomain_owner[source])
                elif source in box_port_owner\
                        or inside.ports[source].kind != PortKind.INPUT:
                    raise ValueError(
                        "A causal schedule requires every box input to be "
                        "wired from a box output or boundary input.")
            dependencies.append(current)
            for port in ports[arity:]:
                target = inside.edges[port]
                if (target in box_port_owner and target not in domain_owner)\
                        or (target not in box_port_owner and inside.ports[
                            target].kind != PortKind.OUTPUT):
                    raise ValueError(
                        "A causal schedule requires every box output to be "
                        "wired to a box input or boundary output.")

        remaining = [set(items) for items in dependencies]
        ready = [i for i, items in enumerate(remaining) if not items]
        order = []
        while ready:
            box_index = ready.pop(0)
            order.append(box_index)
            for target, items in enumerate(remaining):
                if box_index in items:
                    items.remove(box_index)
                    if not items and target not in order\
                            and target not in ready:
                        ready.append(target)
        if len(order) != len(inside.boxes):
            raise ValueError(
                "A causal schedule requires an acyclic box dependency graph.")
        return tuple(order)

    @property
    def indices(self) -> dict:
        """ The routing of the map as index arrays of the backend. """
        return self.inside.indices(self.backend, self.prototype)

    def zeros(self, width: int):
        """ A zero message with the execution's batch size and prototype. """
        return self.backend.zeros(
            self.batch_size, width, like=self.prototype)

    def validate(self, value, width: int, label: str):
        """ Validate the rank, batch size and width of a message tensor. """
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"{label} must have shape (batch_size, {width}).")
        if shape[0] != self.batch_size or shape[1] != width:
            raise ValueError(
                f"{label} has shape {tuple(shape)}, expected "
                f"({self.batch_size}, {width}).")
        return value

    @staticmethod
    def _values(given):
        """ Yield non-null tensors from a tensor or per-item sequence. """
        if isinstance(given, (list, tuple)):
            return (value for value in given if value is not None)
        return iter(()) if given is None else iter((given, ))

    def flat(self, given, widths: tuple[int, ...], label: str):
        """
        One flat array from a flat array, a per-item sequence with ``None``
        for zeros, or nothing at all.

        Parameters:
            given : The messages, or ``None``.
            widths : The width of each item.
            label : The name of the argument, for the error messages.
        """
        if given is None:
            return self.zeros(sum(widths))
        if not isinstance(given, (list, tuple)):
            return self.validate(given, sum(widths), label)
        if len(given) != len(widths):
            raise ValueError(
                f"{label} must contain {len(widths)} messages, "
                f"got {len(given)}.")
        values = tuple(
            self.zeros(width) if value is None
            else self.validate(value, width, f"{label}[{i}]")
            for i, (value, width) in enumerate(zip(given, widths)))
        return self.backend.concatenate(values) if values \
            else self.zeros(0)

    def initialize(self):
        """ Initialize the flat messages and the flat private memory. """
        inside, backend = self.inside, self.backend
        widths = inside.port_widths
        reference = next((
            value for given in (self.x, self.init, self.memory)
            for value in self._values(given)), None)
        if reference is not None:
            shape = getattr(reference, "shape", None)
            if shape is None or len(shape) != 2:
                raise ValueError(
                    "Messages must have shape (batch_size, width).")
        self.batch_size = 1 if reference is None else shape[0]
        self.prototype = reference if reference is not None\
            else backend.prototype(self.modules)
        indices = self.indices

        self.source = self.zeros(sum(widths))
        if self.x is not None:
            self.validate(
                self.x, sum(widths[i] for i in inside.input_ports), "x")
            if inside.input_ports:
                self.source = backend.put(
                    self.source, indices["input"], self.x)
        self.initial = self.flat(self.init, widths, "init")
        self.stored = self.flat(self.memory, inside.memory_widths, "memory")
        self.incoming = self.initial + self.source[:, indices["src"]]
        self.outgoing = None
        return self.incoming

    @property
    def memories(self) -> tuple:
        """ The private memory of each box occurrence, from the flat one. """
        widths = self.inside.memory_widths
        return self.backend.split(self.stored, widths) if widths else ()

    def step(self, incoming, stored, inject: bool) -> tuple:
        """
        One round from the flat state: every box applied, the outputs
        routed along the wires and the initial messages re-added when
        injecting. The step is :func:`make_step`'s closure, cached and
        compiled on the map, see :meth:`CMap.step`.

        Parameters:
            incoming : The flat incoming messages.
            stored : The flat private memory.
            inject : Whether to re-add the initial messages after routing.
        """
        step = self.inside.step(self.backend, self.prototype, self.modules)
        return step(incoming, stored, self.source, self.initial, inject)

    def activate(self):
        """ Apply every box to its messages and private memory. """
        self.outgoing, self.stored = activate(
            self.backend, self.modules, self.indices, self.source,
            self.incoming, self.stored)
        return self.outgoing

    def route(self):
        """ Route the outgoing messages along the edge involution. """
        self.incoming = self.outgoing[:, self.indices["src"]]
        return self.incoming

    def inject(self):
        """ Add the initial messages to the current incoming messages. """
        self.incoming = self.incoming + self.initial
        return self.incoming

    def box_outputs(self, outgoing=None) -> tuple:
        """
        The outgoing messages of each box in its logical port order, or
        ``None`` for each box before any round has run.
        """
        outgoing = self.outgoing if outgoing is None else outgoing
        if outgoing is None:
            return len(self.inside.boxes) * (None, )
        indices = self.indices
        widths = tuple(
            sum(self.inside.port_widths[port] for port in ports)
            for ports in self.inside.routing["boxes"])
        return self.backend.split(
            outgoing[:, indices["ports"]], widths) if widths else ()

    def readout(self, incoming=None, outgoing=None):
        """ Read boundary outputs, or final box outputs for a closed map. """
        incoming = self.incoming if incoming is None else incoming
        if self.inside.has_boundary:
            return incoming[:, self.indices["output"]]\
                if self.inside.output_ports else self.zeros(0)
        return self.box_outputs(outgoing)

    def forward(
            self, n_rounds: int = None, inject: bool = True,
            return_memory: bool = False, return_rounds: bool = False,
            return_flat: bool = False):
        """
        Execute synchronous activation and routing rounds.

        Parameters:
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-add ``init`` after every routing step
                     rather than just before the first round.
            return_memory : Whether to return the final memories, one per
                            box occurrence, beside the result.
            return_rounds : Whether to return the result after every round
                            rather than just the last.
            return_flat : Whether the result is the flat incoming messages
                          of the next round rather than the boundary
                          outputs or the per-box outputs.
        """
        self.initialize()
        n_rounds = len(self.inside.boxes) if n_rounds is None else n_rounds
        if n_rounds < 0:
            raise ValueError("n_rounds cannot be negative.")
        inject = inject and self.init is not None
        rounds = []
        for _ in range(n_rounds):
            self.incoming, self.outgoing, self.stored = self.step(
                self.incoming, self.stored, inject)
            if return_rounds:
                rounds.append(self.incoming if return_flat else self.readout())
        result = rounds if return_rounds else self.incoming if return_flat\
            else self.readout()
        return (result, self.memories) if return_memory else result

    def forward_causal(
            self, inject: bool = True, return_memory: bool = False):
        """
        Execute every box once in topological order, for a feed-forward map.

        Parameters:
            inject : Whether to re-add ``init`` on the ports a box writes.
            return_memory : Whether to return the final memories beside.
        """
        self.initialize()
        backend, indices = self.backend, self.indices
        incoming, outgoing, stored = self.incoming, self.source, self.stored
        for box_index in self.topological_order:
            box = indices["boxes"][box_index]
            public, next_memory = activate_box(
                self.backend, self.modules, box, incoming, stored)
            outgoing = backend.put(outgoing, box["ports"], public)
            if box["memory_width"]:
                stored = backend.put(stored, box["memory"], next_memory)
            arrived = public + self.initial[:, box["targets"]]\
                if inject and self.init is not None else public
            incoming = backend.put(incoming, box["targets"], arrived)
        self.incoming, self.outgoing, self.stored = incoming, outgoing, stored
        result = self.readout()
        return (result, self.memories) if return_memory else result

    __call__ = forward


def activate_box(backend, modules, group: dict, incoming, stored) -> tuple:
    """
    Apply one module to every box of a group at once, returning the public
    outputs
    and the next memories, one row per box and batch, as ``(batch_size,
    n_boxes * width)`` arrays ready to be put back at the group's indices.

    Parameters:
        backend : The execution backend.
        modules : The backend-owned modules, in the map's unique-module
                  order.
        group : An entry of ``groups`` or of ``boxes`` in the map's
                :meth:`~discopy.neural.CMap.indices`.
        incoming : The flat incoming messages.
        stored : The flat private memory.
    """
    width, memory_width = group["width"], group["memory_width"]
    n_boxes, batch_size = len(group["boxes"]), incoming.shape[0]
    values = incoming[:, group["ports"]].reshape(-1, width)
    if memory_width:
        values = backend.concatenate((values, stored[
            :, group["memory"]].reshape(-1, memory_width)))
    outputs = backend.activate(modules[group["module"]], values)
    shape = getattr(outputs, "shape", None)
    if shape is None or tuple(shape) != (
            batch_size * n_boxes, width + memory_width):
        raise ValueError(
            f"output of box {group['boxes'][0]} has shape "
            f"{None if shape is None else tuple(shape)}, expected "
            f"({batch_size * n_boxes}, {width + memory_width}).")
    public, next_memory = backend.split(outputs, (width, memory_width))
    return public.reshape(batch_size, -1), next_memory.reshape(
        batch_size, -1)


def activate(backend, modules, indices: dict, source, incoming, stored):
    """
    Apply every box of a map to its messages and private memory, one call
    per group of boxes sharing a module: the flat outgoing messages, with
    the boundary inputs of ``source`` on the input ports, and the next
    flat memory.

    Parameters:
        backend : The execution backend.
        modules : The backend-owned modules.
        indices : The map's :meth:`~discopy.neural.CMap.indices`.
        source : The flat boundary inputs, zero elsewhere.
        incoming : The flat incoming messages.
        stored : The flat private memory.
    """
    outgoing = source
    for group in indices["groups"]:
        public, next_memory = activate_box(
            backend, modules, group, incoming, stored)
        outgoing = backend.put(outgoing, group["ports"], public)
        if group["memory_width"]:
            stored = backend.put(stored, group["memory"], next_memory)
    return outgoing, stored


def make_step(backend, modules, indices: dict):
    """
    Return one round of message passing as a function of flat arrays
    alone, ``(incoming, stored, source, initial, inject) -> (incoming,
    outgoing, stored)``, so that a backend can compile it once per map:
    the boxes applied by :func:`activate`, the outputs routed by the
    ``src`` permutation and the initial messages re-added when ``inject``.

    Parameters:
        backend : The execution backend.
        modules : The backend-owned modules.
        indices : The map's :meth:`~discopy.neural.CMap.indices`.
    """
    def step(incoming, stored, source, initial, inject: bool):
        outgoing, stored = activate(
            backend, modules, indices, source, incoming, stored)
        incoming = outgoing[:, indices["src"]]
        return (incoming + initial if inject else incoming), outgoing, stored

    return step


def box_forward(inside, messages, backend, modules):
    """
    Run a map as one box of the all-port protocol, the way the wrappers of
    every :class:`~discopy.neural.backend.Backend` do: ``messages`` is one
    batch of incoming messages on the public ports of ``inside`` followed by
    its private memory, and the result is the outgoing messages followed by
    the next memory, as :meth:`Execution.forward` computes them.
    """
    dom_width, cod_width = sum(inside.dom.inside), sum(inside.cod.inside)
    memory_width = sum(inside.memory_widths)
    expected = dom_width + cod_width + memory_width
    shape = getattr(messages, "shape", None)
    if shape is None or len(shape) != 2 or shape[-1] != expected:
        actual = None if shape is None else tuple(shape)
        raise ValueError(
            f"Nested map messages have shape {actual}, "
            f"expected (batch_size, {expected}).")
    inputs, outputs, memory = backend.split(
        messages, (dom_width, cod_width, memory_width))
    execution = Execution(
        inside, memory=memory if memory_width else None,
        backend=backend, modules=modules)
    boundary_ports = inside.input_ports + inside.output_ports
    initial = [None] * inside.n_ports
    for port, value in zip(boundary_ports, backend.split(
            backend.concatenate((inputs, outputs)),
            tuple(inside.port_widths[i] for i in boundary_ports))):
        initial[inside.edges[port]] = value
    execution.init = initial
    execution.forward()
    public = execution.incoming[:, execution.indices["boundary"]]\
        if boundary_ports else backend.zeros(shape[0], 0, like=messages)
    return backend.concatenate((public, execution.stored))
