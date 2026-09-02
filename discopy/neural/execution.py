# -*- coding: utf-8 -*-

"""
The execution formula of the geometry of interaction, on any backend.

:class:`Execution` runs a neural :class:`~discopy.neural.CMap` one Python
call per box per round, on whichever :class:`~discopy.neural.Backend` holds
the tensors: it is the reference the vectorised torch path of
:meth:`CMap.forward <discopy.neural.CMap.forward>` is checked against, the
path every other backend runs, and the one that carries the private
memory of a :class:`~discopy.neural.Network` and the causal schedule of a
feed-forward map.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Execution
"""

from __future__ import annotations

from functools import cached_property

from discopy.cmap import PortKind
from discopy.neural.backend import Backend, get_backend


class Execution:
    """
    The geometry-of-interaction execution of a neural combinatorial map.

    Let ``edge`` be the fixpoint-free involution on ports and ``activate``
    apply every network independently. One synchronous round first activates
    the boxes, then routes their outgoing messages with
    ``incoming[i] = outgoing[edge[i]]``. Boundary inputs are emitted before
    every round, while ``init`` is optionally injected after every routing
    step as well as before the first round.

    Parameters:
        inside : The combinatorial map to execute.
        x : The boundary input.
        init : The initial incoming messages.
        memory : The initial private memory, one tensor per box occurrence.
        backend : The execution backend, the current backend by default.
        modules : The backend-owned modules, in the map's unique-module order.
                  The modules inside the boxes are used by default, so that
                  a backend can train copies of them without rebuilding
                  the map.
    """
    def __init__(
            self, inside, x=None, init=None, memory=None,
            backend: str | Backend = None, modules=None):
        self.inside = inside
        self.x, self.init = x, init
        self.memory = memory
        self.backend = get_backend(backend)
        self.modules = inside.modules if modules is None else tuple(modules)
        if len(self.modules) != len(inside.modules):
            raise ValueError(
                f"Expected {len(inside.modules)} modules, "
                f"got {len(self.modules)}.")
        self.batch_size, self.prototype = 1, None
        self.initial = self.boundary = self.incoming = ()
        self.outgoing = self.box_outputs = ()
        self.memories = ()

    @property
    def input_ports(self) -> tuple[int, ...]:
        """ The indices of boundary input ports. """
        return self.inside.input_ports

    @property
    def output_ports(self) -> tuple[int, ...]:
        """ The indices of boundary output ports. """
        return self.inside.output_ports

    @cached_property
    def box_ports(self) -> tuple[tuple[int, ...], ...]:
        """ The ports of each box in domain-then-codomain order. """
        return tuple(
            self.inside.box_ports(i) for i in range(len(self.inside.boxes)))

    @property
    def memory_widths(self) -> tuple[int, ...]:
        """ The private memory width of each box occurrence. """
        return self.inside.memory_widths

    @cached_property
    def topological_order(self) -> tuple[int, ...]:
        """ Order boxes from boundary inputs towards boundary outputs. """
        if self.inside.loops:
            raise ValueError(
                "A causal schedule requires an acyclic map, without loops.")
        domain_owner, codomain_owner, box_port_owner = {}, {}, {}
        for box_index, (box, ports) in enumerate(zip(
                self.inside.boxes, self.box_ports)):
            arity = len(box.dom)
            for port in ports:
                box_port_owner[port] = box_index
            for port in ports[:arity]:
                domain_owner[port] = box_index
            for port in ports[arity:]:
                codomain_owner[port] = box_index

        dependencies = []
        for box_index, (box, ports) in enumerate(zip(
                self.inside.boxes, self.box_ports)):
            arity, current = len(box.dom), set()
            for port in ports[:arity]:
                source = self.inside.edges[port]
                if source in codomain_owner:
                    current.add(codomain_owner[source])
                elif source in box_port_owner:
                    raise ValueError(
                        "A causal schedule requires every box input to be "
                        "wired from a box output or boundary input.")
                elif self.inside.ports[source].kind != PortKind.INPUT:
                    raise ValueError(
                        "A causal schedule requires every box input to be "
                        "wired from a box output or boundary input.")
            dependencies.append(current)

            for port in ports[arity:]:
                target = self.inside.edges[port]
                if target in box_port_owner and target not in domain_owner:
                    raise ValueError(
                        "A causal schedule requires every box output to be "
                        "wired to a box input or boundary output.")
                if target not in box_port_owner\
                        and self.inside.ports[target].kind != PortKind.OUTPUT:
                    raise ValueError(
                        "A causal schedule requires every box output to be "
                        "wired to a box input or boundary output.")

        remaining = [set(items) for items in dependencies]
        ready = [
            box_index for box_index, items in enumerate(remaining)
            if not items]
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
        if len(order) != len(self.inside.boxes):
            raise ValueError(
                "A causal schedule requires an acyclic box dependency graph.")
        return tuple(order)

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

    def _initialize_messages(self, given, widths, label):
        """ Normalize a tensor or nullable per-item sequence to messages. """
        if given is None:
            values = len(widths) * (None, )
        elif isinstance(given, (list, tuple)):
            if len(given) != len(widths):
                raise ValueError(
                    f"{label} must contain {len(widths)} messages, "
                    f"got {len(given)}.")
            values = given
        else:
            self.validate(given, sum(widths), label)
            values = self.backend.split(given, widths) if widths else ()
        return tuple(
            self.zeros(width) if value is None
            else self.validate(value, width, f"{label}[{i}]")
            for i, (value, width) in enumerate(zip(values, widths)))

    def initialize(self) -> tuple:
        """ Initialize public messages and per-box private memories. """
        widths = self.inside.port_dims
        given = (
            self._values(self.x), self._values(self.init),
            self._values(self.memory))
        reference = next((
            value for values in given for value in values), None)
        if reference is not None:
            shape = getattr(reference, "shape", None)
            if shape is None or len(shape) != 2:
                raise ValueError(
                    "Messages must have shape (batch_size, width).")
        self.batch_size = 1 if reference is None else shape[0]
        self.prototype = reference if reference is not None\
            else self.backend.prototype(self.modules)

        boundary = [self.zeros(width) for width in widths]
        if self.x is not None:
            self.validate(
                self.x, sum(widths[i] for i in self.input_ports), "x")
            slices = self.backend.split(
                self.x, tuple(widths[i] for i in self.input_ports))
            for i, message in zip(self.input_ports, slices):
                boundary[i] = message

        initial = self._initialize_messages(self.init, widths, "init")
        memories = self._initialize_messages(
            self.memory, self.memory_widths, "memory")

        incoming = list(initial)
        for i in self.input_ports:
            edge = self.inside.edges[i]
            incoming[edge] = incoming[edge] + boundary[i]

        self.initial = tuple(initial)
        self.boundary = tuple(boundary)
        self.incoming = tuple(incoming)
        self.outgoing = ()
        self.box_outputs = len(self.inside.boxes) * (None, )
        self.memories = memories
        return self.incoming

    def activate_box(self, box_index: int, incoming):
        """ Apply one network to its public messages and private memory. """
        widths = self.inside.port_dims
        ports = self.box_ports[box_index]
        public_widths = tuple(widths[i] for i in ports)
        memory_width = self.memory_widths[box_index]
        values = tuple(incoming[i] for i in ports)\
            + (self.memories[box_index], )
        module = self.modules[self.inside.module_indices[box_index]]
        output = self.backend.activate(
            module, self.backend.concatenate(values))
        self.validate(
            output, sum(public_widths) + memory_width,
            f"output of box {box_index}")
        chunks = self.backend.split(
            output, public_widths + (memory_width, ))
        public, next_memory = chunks[:-1], chunks[-1]
        box_output = (
            self.backend.concatenate(public) if public else self.zeros(0))
        return public, next_memory, box_output

    def activate(self) -> tuple:
        """ Apply each network to its public messages and private memory. """
        widths = self.inside.port_dims
        outgoing = [self.zeros(width) for width in widths]
        for i in self.input_ports:
            outgoing[i] = self.boundary[i]

        box_outputs, memories = [], []
        for box_index, ports in enumerate(self.box_ports):
            public, next_memory, box_output = self.activate_box(
                box_index, self.incoming)
            box_outputs.append(box_output)
            memories.append(next_memory)
            for i, chunk in zip(ports, public):
                outgoing[i] = chunk

        self.outgoing = tuple(outgoing)
        self.box_outputs = tuple(box_outputs)
        self.memories = tuple(memories)
        return self.outgoing

    def route(self) -> tuple:
        """ Route outgoing messages along the edge involution. """
        self.incoming = tuple(
            self.outgoing[self.inside.edges[i]]
            for i in range(self.inside.n_ports))
        return self.incoming

    def inject(self) -> tuple:
        """ Add the initial messages to the current incoming messages. """
        self.incoming = tuple(
            message + initial
            for message, initial in zip(self.incoming, self.initial))
        return self.incoming

    def readout(self):
        """ Read boundary outputs, or final box outputs for a closed map. """
        if self.inside.has_boundary:
            return self.backend.concatenate(tuple(
                self.incoming[i] for i in self.output_ports))\
                if self.output_ports else self.zeros(0)
        return self.box_outputs

    def forward(
            self, n_rounds: int = None, inject: bool = True,
            return_memory: bool = False):
        """ Execute synchronous activation and routing rounds. """
        self.initialize()
        n_rounds = len(self.inside.boxes) if n_rounds is None else n_rounds
        if n_rounds < 0:
            raise ValueError("n_rounds cannot be negative.")
        for _ in range(n_rounds):
            self.activate()
            self.route()
            if inject and self.init is not None:
                self.inject()
        result = self.readout()
        return (result, self.memories) if return_memory else result

    def forward_causal(
            self, inject: bool = True, return_memory: bool = False):
        """ Execute every box once in topological order. """
        self.initialize()
        widths = self.inside.port_dims
        incoming = list(self.incoming)
        outgoing = [self.zeros(width) for width in widths]
        for port in self.input_ports:
            outgoing[port] = self.boundary[port]
        box_outputs = list(self.box_outputs)
        memories = list(self.memories)

        for box_index in self.topological_order:
            ports = self.box_ports[box_index]
            public, next_memory, box_output = self.activate_box(
                box_index, incoming)
            box_outputs[box_index] = box_output
            memories[box_index] = next_memory
            for port, chunk in zip(ports, public):
                outgoing[port] = chunk
                target = self.inside.edges[port]
                incoming[target] = chunk + self.initial[target]\
                    if inject and self.init is not None else chunk

        self.incoming, self.outgoing = tuple(incoming), tuple(outgoing)
        self.box_outputs, self.memories = (
            tuple(box_outputs), tuple(memories))
        result = self.readout()
        return (result, self.memories) if return_memory else result

    __call__ = forward
