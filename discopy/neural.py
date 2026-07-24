# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, with additive
dimensions as objects and concatenation as tensor.

A :class:`Network` with domain ``Dim(a_1, ..., a_m)`` and codomain
``Dim(b_1, ..., b_n)`` carries one :class:`torch.nn.Module` from ``R ** w``
to ``R ** w`` for ``w = a_1 + ... + a_m + b_1 + ... + b_n``, reading incoming
messages on all its ports and emitting outgoing messages on all its ports.
Networks compose with the cartesian product of vector spaces, so the tensor
of dimensions is their sum with the zero-dimensional space ``Dim(0)`` as
unit; dimensions are self-dual so that cups, caps and swaps are pure
rerouting.

The combinatorial maps of this category are graph neural networks: the
:meth:`CMap.forward` pass does synchronous message passing along the wires,
which implements the execution formula of the geometry of interaction, see
:cite:t:`Abramsky96` and :mod:`discopy.interaction` for the Int-construction
of Joyal, Street & Verity :cite:p:`JoyalEtAl96`.

The forward pass is vectorized: all the messages live in one flat tensor, one
round of routing is a single permutation of its last axis, and every box that
shares a module and a port signature is evaluated in one batched call, so a
grid of identical cells costs one module call per round rather than one per
cell. On a closed map -- no boundary, every port owned by a box -- the
messages are stored in box order rather than port order, so each module reads
its inputs as a contiguous view and one round costs exactly one permutation,
with no gather or scatter around the module calls. It runs on whatever device
its parameters live on, so ``cmap.to("cuda")`` followed by
``cmap(x.to("cuda"))`` trains on the GPU. :meth:`CMap.forward_reference`
is the equivalent one-call-per-box implementation, kept for clarity and tests;
:meth:`CMap.compile` wraps the per-round step in ``torch.compile`` for maps
whose rounds are launch-bound rather than compute-bound.
Cells need not be feedforward: a box can carry recurrent state between rounds
along a self-wired pair of ports, i.e. a feedback loop in the sense of the
trace.

Note that ``import discopy.neural`` does not import ``torch``: networks can
be built, composed and rewired without it, only evaluating their modules
requires it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dim
    Diagram
    Box
    Network
    Cup
    Cap
    Swap
    Functor
    Hypergraph
    CMap

Example
-------

Message passing on the combinatorial map of a diagram computes its image
under the execution formula, e.g. rerouting for a snake:

>>> import torch
>>> snake = Id(Dim(2)).transpose().to_map()
>>> snake.boxes
()
>>> x = torch.tensor([[0.1, 0.2]])
>>> assert (snake(x) == x).all()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import cached_property

from discopy import compact, monoidal
from discopy.cat import ar_factory, ob_factory
from discopy.cmap import PortKind
from discopy.pivotal import Ty
from discopy.utils import assert_isinstance

if TYPE_CHECKING:
    import torch


@ob_factory
class Dim(monoidal.Dim, Ty):
    """
    A dimension is a tuple of positive integers seen as a self-dual type,
    with addition as tensor and the zero-dimensional space as unit.

    Example
    -------
    >>> assert Dim(0) == Dim() and Dim(0) @ Dim(2) @ Dim(3) == Dim(2, 3)
    >>> assert Dim(2, 3).l == Dim(2, 3).r == Dim(3, 2)
    """
    unit = 0
    l = r = property(lambda self: self.ob(*self.inside[::-1]))


@ar_factory
class Diagram(compact.Diagram):
    """
    A neural diagram is a compact diagram with dimensions as objects.

    Parameters:
        inside (Layer) : The layers of the diagram.
        dom (Dim) : The domain of the diagram, i.e. its input.
        cod (Dim) : The codomain of the diagram, i.e. its output.
    """
    ob = Dim


class Box(compact.Box, Diagram):
    """
    A neural box is a compact box between dimensions.

    Parameters:
        name (str) : The name of the box.
        dom (Dim) : The domain of the box, i.e. its input.
        cod (Dim) : The codomain of the box, i.e. its output.
    """


class Cup(compact.Cup, Box):
    """
    A neural cup is a compact cup between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Cap(compact.Cap, Box):
    """
    A neural cap is a compact cap between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Swap(compact.Swap, Box):
    """
    A neural swap is a compact swap between dimensions.

    Parameters:
        left (Dim) : The dimension on the top left and bottom right.
        right (Dim) : The dimension on the top right and bottom left.
    """


class Network(Box):
    """
    A network is a neural box together with a torch module computing it.

    The module maps ``R ** width`` to ``R ** width`` for ``width`` the sum
    of the domain and codomain dimensions, reading one incoming message and
    emitting one outgoing message on every port, in the order given by the
    domain followed by the codomain. Reusing the same network instance, or
    the same module, as several boxes shares its weights.

    Parameters:
        name : The name of the network.
        dom : The domain of the network, i.e. its input.
        cod : The codomain of the network, i.e. its output.
        module : The torch module of the network.

    Note
    ----
    Networks compare equal when they have the same name, shape and module,
    where missing modules compare equal and given modules compare by
    identity. The dagger and rotation of a network reuse its module, with
    the weights read in the new port order.

    Example
    -------
    >>> import torch
    >>> f = Network('f', Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    >>> g = Network('g', Dim(3), Dim(2), module=torch.nn.Linear(5, 5))
    >>> (f >> g).dom == (f >> g).cod == Dim(2)
    True
    >>> f.module(torch.ones(1, 5)).shape
    torch.Size([1, 5])
    >>> assert f[::-1].module is f.module
    """
    def __init__(self, name: str, dom: Dim, cod: Dim,
                 module: "torch.nn.Module" = None, data=None, **params):
        self.module = module if module is not None else data
        super().__init__(name, dom, cod, data=self.module, **params)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __repr__(self):
        return f"neural.Network({self.name!r}, " \
               f"dom={self.dom!r}, cod={self.cod!r})"


class Functor(compact.Functor):
    """
    A neural functor is a compact functor between neural diagrams.

    Parameters:
        ob (Mapping[Dim, Dim]) : Map from atomic :class:`Dim` to `cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


class Hypergraph(compact.Hypergraph):
    """ A neural hypergraph is a compact hypergraph between dimensions. """
    functor = Functor


class CMap(compact.CMap):
    """
    A neural combinatorial map is a compact map with networks as boxes,
    which computes as a graph neural network.

    The :meth:`forward` pass does synchronous message passing: one message
    per port, travelling along the wires given by the ``edges`` involution.
    An optimizer only needs :meth:`parameters` and a training loop only
    needs to call the map, so it can be trained like any torch module;
    :meth:`as_network` wraps it back into a :class:`Network` with a fresh
    module inside, for use inside a larger model.

    Example
    -------
    >>> f = Network('f', Dim(2), Dim(3, 2))
    >>> fm = f.to_map()
    >>> fm.box_ports(0)
    (1, 3, 2)
    >>> fm.port_widths
    (2, 2, 2, 3, 3, 2)
    """
    functor = Functor

    def box_ports(self, index: int) -> tuple[int, ...]:
        """
        The global port indices of a box in logical order, i.e. its domain
        ports followed by its codomain ports, undoing the clockwise order
        which stores the codomain ports reversed.

        Parameters:
            index : The index of the box.
        """
        ports = self._box_port_indices[index]
        arity = len(self.boxes[index].dom)
        return ports[:arity] + tuple(reversed(ports[arity:]))

    @property
    def port_widths(self) -> tuple[int, ...]:
        """ The dimension carried by each port of the map. """
        return tuple(
            sum(getattr(port.obj, "inside", (port.obj, )))
            for port in self.ports)

    @cached_property
    def module_list(self) -> "torch.nn.ModuleList":
        """ The distinct torch modules of the networks inside the map. """
        import torch
        modules, seen = [], set()
        for box in self.boxes:
            assert_isinstance(box, Network)
            if box.module is None:
                raise ValueError(f"{box!r} has no module.")
            if id(box.module) not in seen:
                seen.add(id(box.module))
                modules.append(box.module)
        return torch.nn.ModuleList(modules)

    def parameters(self, recurse: bool = True):
        """ The parameters of the networks inside the map. """
        return self.module_list.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """ The named parameters of the networks inside the map. """
        return self.module_list.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        """ The state dict of the networks inside the map. """
        return self.module_list.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        """ Load a state dict into the networks inside the map. """
        return self.module_list.load_state_dict(state_dict, **kwargs)

    def train(self, mode: bool = True) -> CMap:
        """ Set the networks inside the map to training mode. """
        self.module_list.train(mode)
        return self

    def eval(self) -> CMap:
        """ Set the networks inside the map to evaluation mode. """
        return self.train(False)

    def to(self, *args, **kwargs) -> CMap:
        """ Move the networks inside the map to a device or dtype. """
        self.module_list.to(*args, **kwargs)
        return self

    def as_network(self, name: str = "network") -> Network:
        """
        Wrap the map back into a :class:`Network` with a fresh torch module
        inside, whose forward pass is the message passing of the map. The
        module registers the modules of the networks inside the map, so
        that the result can be used inside a larger model.

        Parameters:
            name : The name of the network.
        """
        import torch
        cmap = self

        class CMapModule(torch.nn.Module):
            """ A combinatorial map wrapped as a torch module. """
            def __init__(self):
                super().__init__()
                self.networks = cmap.module_list

            def forward(self, *args, **kwargs):
                """ Message passing over the wrapped map. """
                return cmap.forward(*args, **kwargs)

        return Network(name, self.dom, self.cod, module=CMapModule())

    def _prepare(self, x, init):
        """
        Common set-up for the two forward passes: the number of rows in the
        batch and the ``dtype``/``device`` of a reference tensor, taken from
        the input, the initial messages or the parameters in that order.
        """
        given = [x] + (list(init)
                       if isinstance(init, (list, tuple)) else [init])
        ref = next((t for t in given if t is not None), None)
        batch_size = 1 if ref is None else ref.shape[0]
        proto = ref if ref is not None else next(
            iter(self.parameters()), None) if self.boxes else None
        kwargs = {} if proto is None else {
            "dtype": proto.dtype, "device": proto.device}
        return batch_size, kwargs

    def forward_reference(self, x: "torch.Tensor" = None, init=None,
                          n_rounds: int = None, inject: bool = True):
        """
        Synchronous message passing along the wires of the map, i.e. the
        execution formula of the geometry of interaction. Reference
        implementation: one Python call per box per round, kept for clarity
        and for testing; see :meth:`forward` for the vectorized equivalent.

        Every port carries one message of shape ``(batch_size, width)``.
        The boundary input ports deliver the corresponding slice of ``x``
        along their wires before the first round and after every round.
        Each round, every network reads the incoming messages on its ports
        and emits outgoing ones, then all messages travel along the wires.

        Parameters:
            x : The input, of shape ``(batch_size, sum of domain widths)``.
            init : The initial incoming messages, given per port or as one
                   tensor of shape ``(batch_size, sum of port widths)``.
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-add ``init`` to the incoming messages at
                     every round rather than just the first.

        Returns:
            The final messages at the boundary output ports, concatenated,
            or the tuple of final per-box outgoing messages in logical port
            order when the map is closed.
        """
        import torch
        ports = self.ports
        widths = self.port_widths
        n_rounds = len(self.boxes) if n_rounds is None else n_rounds
        batch_size, kwargs = self._prepare(x, init)

        def zeros(width):
            return torch.zeros(batch_size, width, **kwargs)

        input_ports = [i for i, port in enumerate(ports)
                       if port.kind == PortKind.INPUT]
        x_slices = dict(zip(input_ports, torch.split(
            x, [widths[i] for i in input_ports], dim=-1))) if x is not None\
            else {i: zeros(widths[i]) for i in input_ports}
        if init is not None:
            if isinstance(init, torch.Tensor):
                init = torch.split(init, list(widths), dim=-1)
            init = [message if message is not None else zeros(width)
                    for message, width in zip(init, widths)]

        incoming = [zeros(width) for width in widths]\
            if init is None else list(init)
        for i in input_ports:
            incoming[self.edges[i]] = incoming[self.edges[i]] + x_slices[i]
        box_outputs = [None] * len(self.boxes)

        for _ in range(n_rounds):
            outgoing = [None] * len(ports)
            for i in input_ports:
                outgoing[i] = x_slices[i]
            for box_index, box in enumerate(self.boxes):
                assert_isinstance(box, Network)
                box_ports = self.box_ports(box_index)
                output = box.module(torch.cat(
                    [incoming[i] for i in box_ports], dim=-1))
                box_outputs[box_index] = output
                for i, chunk in zip(box_ports, torch.split(
                        output, [widths[i] for i in box_ports], dim=-1)):
                    outgoing[i] = chunk
            incoming = [
                outgoing[self.edges[i]]
                if outgoing[self.edges[i]] is not None else zeros(width)
                for i, width in enumerate(widths)]
            if inject and init is not None:
                incoming = [message + initial
                            for message, initial in zip(incoming, init)]

        if len(self.dom) or len(self.cod):
            output_ports = [i for i, port in enumerate(ports)
                            if port.kind == PortKind.OUTPUT]
            return torch.cat([incoming[i] for i in output_ports], dim=-1)\
                if output_ports else zeros(0)
        return tuple(box_outputs)

    @cached_property
    def _routing(self) -> dict:
        """
        Cached index tensors for the vectorized :meth:`forward`, on the CPU
        and moved to the working device at call time:

        * ``total`` : the total width, and ``offsets`` : the flat offset of
          each port,
        * ``src`` : the routing permutation, ``incoming = outgoing[:, src]``,
        * ``input``, ``output`` : the boundary ports of the map,
        * ``groups`` : boxes grouped by shared module and port widths, each
          with the flat indices of their ports in logical order, so that one
          module call evaluates a whole group at once.
        """
        import torch
        widths = self.port_widths
        offsets, total = [], 0
        for width in widths:
            offsets.append(total)
            total += width
        src = torch.empty(total, dtype=torch.long)
        for i, width in enumerate(widths):
            j = self.edges[i]
            src[offsets[i]:offsets[i] + width] = torch.arange(
                offsets[j], offsets[j] + widths[j])

        groups: dict = {}
        for index, box in enumerate(self.boxes):
            assert_isinstance(box, Network)
            box_ports = self.box_ports(index)
            key = (id(box.module), tuple(widths[i] for i in box_ports))
            groups.setdefault(key, (box.module, []))[1].append(
                (index, box_ports))

        def gather(members):
            return torch.tensor([
                [k for i in box_ports
                 for k in range(offsets[i], offsets[i] + widths[i])]
                for _, box_ports in members], dtype=torch.long)

        return {
            "total": total, "offsets": offsets, "src": src,
            "input": tuple(i for i, port in enumerate(self.ports)
                           if port.kind == PortKind.INPUT),
            "output": tuple(i for i, port in enumerate(self.ports)
                            if port.kind == PortKind.OUTPUT),
            "groups": tuple(
                (module, tuple(index for index, _ in members), gather(members))
                for module, members in groups.values())}

    @cached_property
    def _fused_routing(self) -> dict:
        """
        Index tensors for the closed-map fast path of :meth:`forward`, where
        the flat messages are stored in *box order* -- for each group of
        boxes sharing a module, one contiguous block of shape ``(n_boxes,
        width)`` -- rather than in port order:

        * ``layout`` : the port-order position stored at each box-order
          position, i.e. ``box_order = port_order[..., layout]``,
        * ``inverse`` : the inverse permutation, back to port order,
        * ``perm`` : one round of routing in box order, the wire involution
          conjugated by the change of layout, ``inverse . src . layout``,
        * ``metas`` : per group, its module, box indices and block shape.

        In this layout every module reads its inputs as a contiguous view
        and one round of routing is the single permutation ``perm``: the
        clone, the scatters and the input gathers of the port-order step
        all disappear.  Only defined for closed maps, where every port
        belongs to a box, so the group blocks partition the flat tensor.
        """
        import torch
        routing = self._routing
        layout = torch.cat([
            gather.reshape(-1) for _, _, gather in routing["groups"]])
        assert len(layout) == routing["total"], "the map is not closed"
        inverse = torch.empty_like(layout)
        inverse[layout] = torch.arange(len(layout))
        return {
            "layout": layout, "inverse": inverse,
            "perm": inverse[routing["src"][layout]],
            "metas": tuple(
                (module, indices, gather.shape[0], gather.shape[1])
                for module, indices, gather in routing["groups"])}

    def _device_routing(self, device) -> dict:
        """
        The index tensors of :attr:`_routing` -- and, for closed maps, of
        :attr:`_fused_routing` -- on a device, cached so that repeated
        forward passes do not re-copy them from the CPU.
        """
        cache = self.__dict__.setdefault("_device_routing_cache", {})
        if device not in cache:
            routing = self._routing
            entry = {
                "src": routing["src"].to(device),
                "groups": tuple(
                    (module, indices, gather.to(device))
                    for module, indices, gather in routing["groups"])}
            if not routing["input"] and not routing["output"] and self.boxes:
                fused = self._fused_routing
                entry.update(
                    layout=fused["layout"].to(device),
                    inverse=fused["inverse"].to(device),
                    perm=fused["perm"].to(device),
                    metas=fused["metas"])
            cache[device] = entry
        return cache[device]

    def compile(self, unroll: bool = False, **kwargs) -> CMap:
        """
        Compile the per-round step of :meth:`forward` with ``torch.compile``.

        Message passing over a small map is launch-bound rather than
        compute-bound: each round issues many small kernels whose launch
        overhead dwarfs their arithmetic. Compiling the round step fuses
        them, typically a several-fold wall-clock speedup on a GPU. The
        round loop stays in Python, so ``n_rounds`` stays dynamic and does
        not trigger recompilation; compilation happens lazily on the first
        forward pass per device and batch size.

        Parameters:
            unroll : Whether to compile the whole ``n_rounds`` loop of a
                     closed map as one function rather than one round at a
                     time, so that e.g. ``mode="reduce-overhead"`` replays a
                     run as a single CUDA graph. One graph is compiled per
                     distinct ``n_rounds``, so this suits loops whose depth
                     is fixed; ``return_rounds`` falls back to the
                     round-by-round step.
            kwargs : Passed through to ``torch.compile``, e.g. ``mode``.

        Note
        ----
        Compiled kernels may reorder floating-point reductions, so results
        can differ from the eager path by rounding error (relative
        differences of about ``1e-6``); gradients agree to the same order.
        """
        self._step_compile = kwargs
        self._unroll = unroll
        self.__dict__.pop("_step_cache", None)
        self.__dict__.pop("_step_flat_cache", None)
        self.__dict__.pop("_runner_cache", None)
        return self

    def _step_body(self, device):
        """
        One round of message passing as a single function of flat tensors,
        ``(incoming, source, init) -> (incoming, group outputs)``, cached
        per device. On a closed map the flat tensors are in the box-order
        layout of :attr:`_fused_routing` and ``source`` is ignored; on an
        open map they are in port order.
        """
        import torch
        cache = self.__dict__.setdefault("_step_body_cache", {})
        if device not in cache:
            routing = self._device_routing(device)
            if "perm" in routing:
                perm, metas = routing["perm"], routing["metas"]

                def step(incoming, source, init):
                    chunks, group_outputs, offset = [], [], 0
                    for module, _, n_boxes, width in metas:
                        block = n_boxes * width
                        outputs = module(
                            incoming[:, offset:offset + block]
                            .reshape(-1, width))
                        group_outputs.append(
                            outputs.reshape(-1, n_boxes, width))
                        chunks.append(outputs.reshape(-1, block))
                        offset += block
                    incoming = torch.cat(chunks, dim=-1)[:, perm]
                    if init is not None:
                        incoming = incoming + init
                    return incoming, group_outputs
            else:
                src, groups = routing["src"], routing["groups"]

                def step(incoming, source, init):
                    outgoing = source.clone()
                    group_outputs = []
                    for module, _, gather in groups:
                        n_boxes, width = gather.shape
                        outputs = module(
                            incoming[:, gather.reshape(-1)]
                            .reshape(-1, width)).reshape(-1, n_boxes, width)
                        outgoing[:, gather.reshape(-1)] = outputs.reshape(
                            outputs.shape[0], n_boxes * width)
                        group_outputs.append(outputs)
                    incoming = outgoing[:, src]
                    if init is not None:
                        incoming = incoming + init
                    return incoming, group_outputs
            cache[device] = step
        return cache[device]

    def _compiled(self, function):
        """
        The function wrapped in ``torch.compile`` when :meth:`compile` was
        called on the map, the function itself otherwise.
        """
        if getattr(self, "_step_compile", None) is None:
            return function
        import torch
        return torch.compile(function, **self._step_compile)

    def _step(self, device):
        """
        The round step of :meth:`_step_body`, wrapped in ``torch.compile``
        when :meth:`compile` was called on the map, cached per device.
        """
        cache = self.__dict__.setdefault("_step_cache", {})
        if device not in cache:
            cache[device] = self._compiled(self._step_body(device))
        return cache[device]

    def _step_flat(self, device):
        """
        The round step of :meth:`_step_body` followed by the permutation
        back to port order, ``(incoming, init) -> (incoming, flat, group
        outputs)``, cached per device and compiled like :meth:`_step`.
        This is the step of ``return_rounds`` with ``return_flat`` on a
        closed map: emitting the port-order flat inside the step keeps
        the per-round read out of the eager path.
        """
        cache = self.__dict__.setdefault("_step_flat_cache", {})
        if device not in cache:
            body = self._step_body(device)
            inverse = self._device_routing(device)["inverse"]

            def step(incoming, init):
                incoming, group_outputs = body(incoming, None, init)
                return incoming, incoming[:, inverse], group_outputs

            cache[device] = self._compiled(step)
        return cache[device]

    def _runner(self, device, n_rounds: int):
        """
        The whole ``n_rounds`` loop over :meth:`_step_body` as one function,
        wrapped in ``torch.compile`` when :meth:`compile` was called with
        ``unroll=True``, cached per device and number of rounds.
        """
        cache = self.__dict__.setdefault("_runner_cache", {})
        if (device, n_rounds) not in cache:
            step = self._step_body(device)

            def run(incoming, init):
                group_outputs = None
                for _ in range(n_rounds):
                    incoming, group_outputs = step(incoming, None, init)
                return incoming, group_outputs

            cache[device, n_rounds] = self._compiled(run)
        return cache[device, n_rounds]

    def forward(self, x: "torch.Tensor" = None, init=None,
                n_rounds: int = None, inject: bool = True,
                return_rounds: bool = False, return_flat: bool = False):
        """
        Vectorized synchronous message passing, equivalent to
        :meth:`forward_reference`: all the messages live in one flat tensor
        of shape ``(batch_size, total width)``, routing along the wires is
        one permutation of the last axis and all the boxes sharing a module
        and a port signature are evaluated in one batched call. It runs on
        the device of the input, the initial messages or the parameters.

        Parameters:
            x : The input, of shape ``(batch_size, sum of domain widths)``.
            init : The initial incoming messages, given per port or as one
                   tensor of shape ``(batch_size, sum of port widths)``.
            n_rounds : The number of rounds, the number of boxes by default.
            inject : Whether to re-add ``init`` to the incoming messages at
                     every round rather than just the first.
            return_rounds : Whether to return the result after every round
                            rather than just the last, e.g. so that a loss
                            can supervise every round of message passing.
            return_flat : Whether to return the flat incoming messages of
                          the next round -- one tensor of shape
                          ``(batch_size, sum of port widths)`` in port
                          order -- instead of slicing the boundary ports or
                          collecting the per-box outputs. This is the same
                          tensor that re-routing the per-box outputs would
                          rebuild, so reading a family of ports from it
                          directly skips that round-trip and keeps the
                          backward graph small; see :meth:`compile`.

        Note
        ----
        On a closed map the messages are held in the box-order layout of
        :attr:`_fused_routing` -- module inputs are contiguous views, one
        round of routing is one permutation -- and are only permuted back
        to port order where the caller sees them, so the results are
        identical to the port-order path element for element.
        """
        import torch
        routing = self._routing
        widths, offsets = self.port_widths, routing["offsets"]
        n_rounds = len(self.boxes) if n_rounds is None else n_rounds
        batch_size, kwargs = self._prepare(x, init)
        device = kwargs.get("device", None)
        step = self._step(device)
        device_routing = self._device_routing(device)
        groups = device_routing["groups"]
        fused = "perm" in device_routing
        closed = not (len(self.dom) or len(self.cod))

        if isinstance(init, (list, tuple)):
            init = torch.cat([
                message if message is not None
                else torch.zeros(batch_size, width, **kwargs)
                for message, width in zip(init, widths)], dim=-1)

        if fused:
            source = None
            incoming = torch.zeros(batch_size, routing["total"], **kwargs)\
                if init is None else init[:, device_routing["layout"]]
            injected = incoming if (inject and init is not None) else None
        else:
            src = device_routing["src"]
            source = torch.zeros(batch_size, routing["total"], **kwargs)
            if x is not None:
                for i, chunk in zip(routing["input"], torch.split(
                        x, [widths[i] for i in routing["input"]], dim=-1)):
                    source[:, offsets[i]:offsets[i] + widths[i]] = chunk
            incoming = source[:, src] if init is None \
                else init + source[:, src]
            injected = init if (inject and init is not None) else None

        def read(incoming, group_outputs):
            if return_flat:
                return incoming[:, device_routing["inverse"]] if fused \
                    else incoming
            if closed:
                box_outputs = [None] * len(self.boxes)
                for (_, indices, _), outputs in zip(groups, group_outputs):
                    if outputs is not None:
                        for k, index in enumerate(indices):
                            box_outputs[index] = outputs[:, k]
                return tuple(box_outputs)
            return torch.cat([
                incoming[:, offsets[i]:offsets[i] + widths[i]]
                for i in routing["output"]], dim=-1)\
                if routing["output"] else torch.zeros(
                    batch_size, 0, **kwargs)

        if fused and n_rounds and not return_rounds \
                and getattr(self, "_unroll", False):
            incoming, group_outputs = self._runner(device, n_rounds)(
                incoming, injected)
            return read(incoming, group_outputs)

        rounds, group_outputs = [], [None] * len(groups)
        if fused and return_rounds and return_flat:
            step_flat = self._step_flat(device)
            for _ in range(n_rounds):
                incoming, flat, group_outputs = step_flat(incoming, injected)
                rounds.append(flat)
            return rounds
        for _ in range(n_rounds):
            incoming, group_outputs = step(incoming, source, injected)
            if return_rounds:
                rounds.append(read(incoming, group_outputs))
        return rounds if return_rounds else read(incoming, group_outputs)

    __call__ = forward


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
