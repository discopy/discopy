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

The forward pass is the :class:`~discopy.neural.execution.Execution` of the
map on a :class:`~discopy.neural.backend.Backend`, torch or JAX: all the
messages live in one flat array, one round of routing is a single
permutation of its last axis, and every box that shares a module and a port
signature is evaluated in one batched call, so a grid of identical cells
costs one module call per round rather than one per cell. It runs on
whatever device its parameters live on, so ``cmap.to("cuda")`` followed by
``cmap(x.to("cuda"))`` trains on the GPU, and :meth:`CMap.compile` hands the
per-round step to the backend's compiler for maps whose rounds are
launch-bound rather than compute-bound.
Cells need not be feedforward: a box can carry state between rounds along a
self-wired pair of ports.  Structurally that pair *is* the categorical trace
of the compact target -- it is wiring, which a functor preserves strictly --
while what it computes over finitely many rounds is delayed feedback: what a
box writes on one end it reads on the other one round later.  Repeated
rounds are the finite iteration ``T ** n``, never a fixed-point solve; see
:mod:`discopy.neural.map` for the transition ``T`` and for the four notions
kept apart there.

This module is the *category*.  Training a neural interpretation of a
diagram goes through :class:`~discopy.neural.MapNN`, which compiles a
diagram and a family of shared generator modules into a :class:`CMap` and
addresses its flat state by ``(generator name, role)`` through
:meth:`CMap.read` and :meth:`CMap.write`.

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
    Network
    Cup
    Cap
    Permutation
    Swap
    Functor
    Hypergraph
    Para
    CMap

Example
-------

Message passing on the combinatorial map of a diagram computes its image
under the execution formula, e.g. rerouting for a snake:

>>> import torch  # doctest: +EXTRA
>>> snake = Id(Dim(2)).transpose().to_map()
>>> snake.boxes
()
>>> x = torch.tensor([[0.1, 0.2]])
>>> assert (snake(x) == x).all()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import cached_property

from discopy import cat, cmap, compact, hypergraph, monoidal, para
from discopy.cat import factory
from discopy.cmap import PortKind
from discopy.neural.backend import Backend, get_backend
from discopy.neural.execution import Execution, make_step
from discopy.pivotal import Ty
from discopy.utils import assert_isinstance, factory_name, from_tree as decode

if TYPE_CHECKING:
    import torch


@factory
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
    l = r = property(lambda self: self.factory(*self.inside[::-1]))
    z = property(lambda self: 0)

    def unwind(self) -> "Dim":
        """ Dimensions are self-dual so their winding is trivial. """
        return self

    def __init__(self, *inside: int, dom=None, cod=None, _scan=True,
                 **kwargs):
        inside = kwargs.pop('inside', inside)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}.")
        for dim in inside:
            assert_isinstance(dim, int)
            if dim < self.unit:
                raise ValueError
        inside = tuple(dim for dim in inside if dim != self.unit)
        white = monoidal.white
        cat.FreeCategory.__init__(
            self, inside, white if dom is None else dom,
            white if cod is None else cod, _scan=False)
        cat.Ob.__init__(self, type(self).__name__)

    def __repr__(self):
        return f"Dim({', '.join(map(repr, self.inside)) or repr(self.unit)})"

    __str__ = __repr__

    def to_tree(self) -> dict:
        return {'factory': factory_name(type(self)),
                'inside': list(self.inside)}

    @classmethod
    def from_tree(cls, tree: dict) -> "Dim":
        return cls(*tree['inside'])


@factory
class Diagram(compact.Diagram):
    """
    A neural diagram is a compact diagram with dimensions as objects.

    Parameters:
        inside (Layer) : The layers of the diagram.
        dom (Dim) : The domain of the diagram, i.e. its input.
        cod (Dim) : The codomain of the diagram, i.e. its output.
    """
    ob = Dim

    def to_map(self) -> "CMap":
        """ Translate a neural diagram into a neural combinatorial map. """
        return CMap.from_diagram(self)


class Network(compact.Box, Diagram):
    """
    A network is a neural box together with a backend module computing it.

    A network is a cell of a message-passing network rather than a
    feedforward layer: its module maps ``R ** width`` to ``R ** width`` for
    ``width`` the sum of the domain, codomain and private memory
    dimensions, i.e. it reads one incoming message and emits one outgoing
    message on every public port at once, in the order given by the domain
    followed by the codomain, then reads the previous memory and emits the
    next one. A feedforward layer is the special case of a module which
    ignores the messages incoming on its codomain, executed with
    :meth:`CMap.forward` and ``causal=True`` so that every box fires once in
    topological order. Reusing the same network instance, or the same
    module, as several boxes shares its weights but each box occurrence has
    its own memory.

    Cups, caps and swaps are networks with ``module`` left to ``None``,
    since they are pure rerouting.

    Parameters:
        name : The name of the network.
        dom : The domain of the network, i.e. its input.
        cod : The codomain of the network, i.e. its output.
        module : The backend-owned module of the network.
        mem : The private memory dimension, empty by default.

    Note
    ----
    Networks compare equal when they have the same name, shape, memory and
    module, where missing modules compare equal and given modules compare
    by identity. The dagger and rotation of a network reuse its module and
    preserve its memory, with the public ports read in the new order. The
    repr omits the module, which has no eval-able representation, so the
    transparency rule ``eval(repr(x)) == x`` holds for a network without
    one.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> f = Network('f', Dim(2), Dim(3), module=torch.nn.Linear(5, 5))
    >>> g = Network('g', Dim(3), Dim(2), module=torch.nn.Linear(5, 5))
    >>> (f >> g).dom == (f >> g).cod == Dim(2)
    True
    >>> f.module(torch.ones(1, 5)).shape
    torch.Size([1, 5])
    >>> assert f[::-1].module is f.module
    """
    module, mem = None, Dim()

    def __init__(self, name: str, dom: Dim, cod: Dim,
                 module: object = None, mem: Dim = Dim(),
                 data=None, **params):
        assert_isinstance(mem, Dim)
        self.mem = mem
        self.module = module if module is not None else data
        super().__init__(name, dom, cod, data=self.module, **params)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __repr__(self):
        mem = f", mem={self.mem!r}" if self.mem else ""
        return f"{factory_name(type(self))}({self.name!r}, {self.dom!r}, "\
            f"{self.cod!r}{mem})"

    def dagger(self) -> Network:
        """ Reverse the public ports, keeping the module and the memory. """
        return type(self)(
            self.name, dom=self.cod, cod=self.dom, module=self.module,
            mem=self.mem, is_dagger=not self.is_dagger, z=self.z)

    def rotate(self, left=False) -> Network:
        """ Rotate the public ports, keeping the module and the memory. """
        del left
        return type(self)(
            self.name, dom=self.cod.r, cod=self.dom.r, module=self.module,
            mem=self.mem, is_dagger=self.is_dagger, z=(self.z + 1) % 2)

    def setoid(self):
        """ Compare given modules by identity and include the memory. """
        result = super().setoid()
        module = None if self.module is None else id(self.module)
        return result[:5] + (module, ) + result[6:] + (self.mem, )

    def to_tree(self) -> dict:
        """ Serialise the shape of the network, memory included. """
        tree = super().to_tree()
        tree['mem'] = self.mem.to_tree()
        if self.z:
            tree['z'] = self.z
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Network:
        """ Deserialise a network, accepting trees without a memory. """
        dom, cod = map(decode, (tree['dom'], tree['cod']))
        mem = decode(tree['mem']) if 'mem' in tree else Dim()
        return cls(
            tree['name'], dom, cod, data=tree.get('data'), mem=mem,
            is_dagger='is_dagger' in tree, z=tree.get('z', 0))


class Cup(compact.Cup, Network):
    """
    A neural cup is a compact cup between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Cap(compact.Cap, Network):
    """
    A neural cap is a compact cap between self-dual dimensions.

    Parameters:
        left (Dim) : The atomic dimension.
        right (Dim) : Its reverse.
    """


class Permutation(compact.Permutation, Network):
    """
    A neural permutation is a compact permutation between dimensions.

    Parameters:
        dom (Dim) : The dimensions to permute.
        perm : The list sending each input to its output.
    """


class Swap(Permutation, compact.Swap, Network):
    """
    A neural swap is a compact swap between dimensions.

    Parameters:
        left (Dim) : The dimension on the top left and bottom right.
        right (Dim) : The dimension on the top right and bottom left.
    """


class Functor(compact.Functor):
    """
    A neural functor is a compact functor between neural diagrams.

    Parameters:
        ob_map (Mapping[Dim, Dim]) : Map from atomic :class:`Dim` to `cod.ob`.
        ar_map (Mapping[Network, Diagram]) : Map from :class:`Network` to
            :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram


Hypergraph = hypergraph.Hypergraph[Diagram]


class Para(para.Compact):
    """
    A parametric network is a network whose weights are boundary values
    rather than hidden inside its modules, i.e. a parametric map
    ``inside : dom @ param -> cod`` over :class:`Diagram` with ``param``
    the dimension of the weights, see :mod:`discopy.para`. Composition and
    tensor accumulate the parameter spaces of the layers and route them to
    the right, so assembling a model does not whisker each layer with the
    weights of all the others.

    Example
    -------
    >>> linear = lambda n: Para(Dim(n), Dim(n), Network(
    ...     f"linear{n}", Dim(n, n * n), Dim(n)), Dim(n * n))
    >>> network = linear(2) >> linear(2)
    >>> network.dom, network.cod, network.param
    (Dim(2), Dim(2), Dim(4, 4))
    >>> assert network.inside == linear(2).inside @ Dim(4) >> linear(2).inside
    >>> assert Para.lift(Diagram.id(Dim(2))) == Para.id(Dim(2))
    """
    category = Diagram


Equation = compact.Equation


def box_ports(cmap, index: int) -> tuple[int, ...]:
    """
    The global port indices of a box in logical order, i.e. its domain
    ports followed by its codomain ports, undoing the clockwise order
    which stores the codomain ports reversed.

    Defined for a map in any compact category, so that the interpretation
    of :mod:`discopy.neural.map` can read the source and the image the same
    way.

    Parameters:
        cmap : The map to read.
        index : The index of the box.

    Example
    -------
    >>> f = Network('f', Dim(2, 3), Dim(4, 5, 6))
    >>> box_ports(f.to_map(), 0)
    (2, 3, 6, 5, 4)
    """
    ports = cmap._box_port_indices[index]
    arity = len(cmap.boxes[index].dom)
    return ports[:arity] + tuple(reversed(ports[arity:]))


def from_wiring(cls, boxes: tuple, wires) -> "CMap":
    """
    A closed map of class ``cls`` given by boxes and wires between pairs
    ``(box_index, port_position)``, where the position counts the
    domain ports of the box followed by its codomain ports.

    Parameters:
        cls : The :class:`~discopy.cmap.CMap` subclass to build.
        boxes : The boxes of the map.
        wires : Pairs of ``(box_index, port_position)`` pairs.

    Raises:
        ValueError : If a port is left unwired or wired twice.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box, CMap
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x @ x), Box('g', x @ x, x)
    >>> cm = from_wiring(CMap, (f, g), [
    ...     ((0, 0), (1, 2)), ((0, 1), (1, 0)), ((0, 2), (1, 1))])
    >>> assert cm.edges.is_fixpoint_free_involution()
    >>> from_wiring(CMap, (f, ), [((0, 0), (0, 0))])
    Traceback (most recent call last):
        ...
    ValueError: Port (0, 0) is wired to itself.
    """
    boxes = tuple(boxes)
    starts, n_ports = [], 0
    for box in boxes:
        starts.append(n_ports)
        n_ports += len(box.dom) + len(box.cod)

    def global_index(box_index: int, position: int) -> int:
        box = boxes[box_index]
        arity, coarity = len(box.dom), len(box.cod)
        if not 0 <= position < arity + coarity:
            raise ValueError(
                f"Box {box_index} has no port {position}.")
        if position < arity:
            return starts[box_index] + position
        return starts[box_index] + arity\
            + (coarity - 1 - (position - arity))

    pairs, seen = [], set()
    for (one, other) in wires:
        i, j = global_index(*one), global_index(*other)
        if i == j:
            raise ValueError(f"Port {one} is wired to itself.")
        for port, position in ((i, one), (j, other)):
            if port in seen:
                raise ValueError(f"Port {position} is wired twice.")
        seen.update((i, j))
        pairs.append((i, j))
    if len(seen) != n_ports:
        missing = sorted(set(range(n_ports)) - seen)
        raise ValueError(f"Ports {missing} are left unwired.")
    edges = cmap.Permutation.from_transpositions(pairs, n_ports)
    return cls(cls.ob(), cls.ob(), boxes, edges)


class CMap(cmap.CMap[Diagram]):
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
    category = Diagram
    functor = Functor

    from_wiring = classmethod(from_wiring)

    def box_ports(self, index: int) -> tuple[int, ...]:
        """
        The global port indices of a box in logical order; see
        :func:`box_ports`.

        Parameters:
            index : The index of the box.
        """
        return box_ports(self, index)

    @cached_property
    def port_widths(self) -> tuple[int, ...]:
        """
        The dimension carried by each port of the map.

        Cached like :attr:`module_list` beside it, and for the same
        reason: it is a function of the boxes, which a map fixes in its
        constructor, and :meth:`forward` reads it on every call.  One
        round of a fixed-point iteration is one call, so a residual curve
        over a diagram with a box per pair rebuilt every port of it per
        round.
        """
        return tuple(
            sum(getattr(port.obj, "inside", (port.obj, )))
            for port in self.ports)

    @cached_property
    def modules(self) -> tuple:
        """ The distinct modules of the networks inside the map. """
        modules, seen = [], set()
        for box in self.boxes:
            assert_isinstance(box, Network)
            if box.module is None:
                raise ValueError(f"{box!r} has no module.")
            if id(box.module) not in seen:
                seen.add(id(box.module))
                modules.append(box.module)
        return tuple(modules)

    @cached_property
    def module_indices(self) -> tuple[int, ...]:
        """ The index in :attr:`modules` of each box occurrence's module. """
        indices = {id(module): i for i, module in enumerate(self.modules)}
        return tuple(indices[id(box.module)] for box in self.boxes)

    @property
    def memory_widths(self) -> tuple[int, ...]:
        """ The private memory width of each box occurrence. """
        for box in self.boxes:
            assert_isinstance(box, Network)
        return tuple(sum(box.mem.inside) for box in self.boxes)

    @property
    def input_ports(self) -> tuple[int, ...]:
        """ The indices of the boundary input ports. """
        return tuple(i for i, port in enumerate(self.ports)
                     if port.kind == PortKind.INPUT)

    @property
    def output_ports(self) -> tuple[int, ...]:
        """ The indices of the boundary output ports. """
        return tuple(i for i, port in enumerate(self.ports)
                     if port.kind == PortKind.OUTPUT)

    @property
    def has_boundary(self) -> bool:
        """ Whether the map has any boundary port. """
        return bool(len(self.dom) or len(self.cod))

    @cached_property
    def module_list(self) -> "torch.nn.ModuleList":
        """ The distinct torch modules of the networks inside the map. """
        import torch
        return torch.nn.ModuleList(self.modules)

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

    def as_network(self, name: str = "network",
                   backend: str | Backend = None) -> Network:
        """
        Wrap the map back into a :class:`Network` with a fresh backend
        module inside, whose forward pass is the message passing of the
        map. The module registers the modules of the networks inside the
        map, so that the result can be trained or nested inside a larger
        model, and its private memory is the memory of every box occurrence
        concatenated.

        Parameters:
            name : The name of the network.
            backend : The backend name or instance, the current one by
                      default.
        """
        backend = get_backend(backend)
        return Network(name, self.dom, self.cod, module=backend.wrap(self),
                       mem=Dim(sum(self.memory_widths)))

    @cached_property
    def routing(self) -> dict:
        """
        The wiring of the map as flat positions, with no tensor framework:

        * ``total`` : the total width, and ``offsets`` : the flat offset of
          each port,
        * ``src`` : the routing permutation, ``incoming = outgoing[src]``,
        * ``input``, ``output`` : the flat positions of the boundary ports,
        * ``boxes`` : the ports of each box in logical order, and
          ``memory`` : the flat range of each box's private memory,
        * ``groups`` : the boxes grouped by module, port widths and memory
          width, each with its ``ports`` and ``memory`` positions in box
          order, so that one module call evaluates a whole group at once.

        Example
        -------
        >>> f = Network('f', Dim(0), Dim(1, 1), module=object())
        >>> ring = CMap(CMap.ob(), CMap.ob(), (f, f), [3, 2, 1, 0])
        >>> ring.routing["src"], ring.routing["boxes"]
        ((3, 2, 1, 0), ((1, 0), (3, 2)))
        >>> ring.routing["groups"][0]["ports"]
        (1, 0, 3, 2)
        """
        widths, memory_widths = self.port_widths, self.memory_widths
        offsets, total = [], 0
        for width in widths:
            offsets.append(total)
            total += width
        memory_offsets = [0]
        for width in memory_widths:
            memory_offsets.append(memory_offsets[-1] + width)

        def flat(ports):
            return tuple(k for i in ports
                         for k in range(offsets[i], offsets[i] + widths[i]))

        boxes = tuple(
            self.box_ports(index) for index in range(len(self.boxes)))
        memory = tuple(
            tuple(range(memory_offsets[i], memory_offsets[i + 1]))
            for i in range(len(self.boxes)))
        groups: dict = {}
        for index, ports in enumerate(boxes):
            key = (self.module_indices[index],
                   tuple(widths[i] for i in ports), memory_widths[index])
            groups.setdefault(key, []).append(index)
        return {
            "total": total, "offsets": tuple(offsets),
            "src": flat(tuple(self.edges)),
            "input": flat(self.input_ports), "output": flat(self.output_ports),
            "boxes": boxes, "memory": memory,
            "groups": tuple({
                "module": module, "boxes": tuple(members),
                "width": sum(box_widths), "memory_width": memory_width,
                "ports": tuple(k for i in members for k in flat(boxes[i])),
                "memory": tuple(k for i in members for k in memory[i])}
                for (module, box_widths, memory_width), members
                in groups.items())}

    def indices(self, backend: Backend, like=None) -> dict:
        """
        The :attr:`routing` as index arrays of a backend, cached per
        backend and device: ``src``, ``input``, ``output``, the
        ``boundary`` inputs then outputs, every box's ``ports`` in box
        order, the ``groups`` and each box on its own in ``boxes``, with
        the ``targets`` its outputs arrive at.

        Parameters:
            backend : The execution backend.
            like : An array on the device the indices should live on.
        """
        key = (backend, getattr(like, "device", None))
        cache = self.__dict__.setdefault("index_cache", {})
        if key not in cache:
            routing, widths = self.routing, self.port_widths

            def index(positions):
                return backend.index(tuple(positions), like)

            def entry(group):
                return dict(group, ports=index(group["ports"]),
                            memory=index(group["memory"]))

            def box(i, ports):
                return entry({
                    "module": self.module_indices[i], "boxes": (i, ),
                    "width": sum(widths[port] for port in ports),
                    "memory_width": self.memory_widths[i],
                    "ports": [k for port in ports for k in range(
                        routing["offsets"][port],
                        routing["offsets"][port] + widths[port])],
                    "memory": routing["memory"][i],
                    "targets": [k for port in ports for k in range(
                        routing["offsets"][self.edges[port]],
                        routing["offsets"][self.edges[port]]
                        + widths[port])]})

            cache[key] = {
                "src": index(routing["src"]),
                "input": index(routing["input"]),
                "output": index(routing["output"]),
                "boundary": index(routing["input"] + routing["output"]),
                "ports": index(k for group in routing["boxes"]
                               for port in group for k in range(
                                   routing["offsets"][port],
                                   routing["offsets"][port] + widths[port])),
                "groups": tuple(map(entry, routing["groups"])),
                "boxes": tuple(
                    box(i, ports) for i, ports in enumerate(routing["boxes"]))}
            for box_entry in cache[key]["boxes"]:
                box_entry["targets"] = index(box_entry["targets"])
        return cache[key]

    def __getstate__(self):
        """ The map without its caches of index arrays and round steps. """
        state = dict(self.__dict__)
        state.pop("index_cache", None)
        state.pop("step_cache", None)
        return state

    def compile(self, **kwargs) -> CMap:
        """
        Compile the per-round :meth:`step` with the backend's compiler,
        ``torch.compile`` on torch, so that the many small kernels of a
        round on a small map are fused. The round loop stays in Python, so
        ``n_rounds`` stays dynamic; compilation happens lazily on the first
        forward pass per backend, device and modules.

        Parameters:
            kwargs : Passed through to the compiler, e.g. ``mode``.
        """
        self.compile_kwargs = kwargs
        self.__dict__.pop("step_cache", None)
        return self

    def step(self, backend: Backend, like=None, modules=None):
        """
        One round of message passing as a function of flat arrays,
        :func:`~discopy.neural.execution.make_step`'s closure over the
        :meth:`indices` and the modules, cached per backend, device and
        modules and compiled when :meth:`compile` was called.

        Parameters:
            backend : The execution backend.
            like : An array on the device the round runs on.
            modules : The backend-owned modules, those of the boxes by
                      default.
        """
        modules = self.modules if modules is None else modules
        key = (backend, getattr(like, "device", None), tuple(map(id, modules)))
        cache = self.__dict__.setdefault("step_cache", {})
        if key not in cache:
            step = make_step(backend, modules, self.indices(backend, like))
            kwargs = getattr(self, "compile_kwargs", None)
            cache[key] = step if kwargs is None\
                else backend.compile(step, **kwargs)
        return cache[key]

    def zeros(self, rows: int, like=None, backend: str | Backend = None):
        """
        An all-zero flat state of ``rows`` rows, one summand per port.

        Parameters:
            rows : The batch size.
            like : An array whose dtype and device the state follows.
            backend : The backend name or instance, the current one by
                      default.
        """
        return get_backend(backend).zeros(
            rows, self.routing["total"], like=like)

    def read(self, state, ports: tuple[int, ...],
             backend: str | Backend = None):
        """
        The messages of a family of equally wide ports, as an array of
        shape ``(rows, len(ports), width)``, from a flat state.

        Parameters:
            state : The flat messages, ``(rows, total)``.
            ports : The global port indices.
            backend : The backend name or instance, the current one by
                      default.
        """
        widths, offsets = self.port_widths, self.routing["offsets"]
        width = widths[ports[0]] if ports else 0
        if any(widths[port] != width for port in ports):
            raise ValueError(
                "ports of different widths cannot be read as one block")
        index = get_backend(backend).index(tuple(
            k for port in ports
            for k in range(offsets[port], offsets[port] + width)), state)
        return state[:, index].reshape(state.shape[0], len(ports), width)

    def write(self, state, ports: tuple[int, ...], values,
              backend: str | Backend = None):
        """
        A copy of a flat state with ``values`` written on a family of
        equally wide ports.

        Parameters:
            state : The flat messages, ``(rows, total)``.
            ports : The global port indices.
            values : An array of shape ``(rows, len(ports), width)``.
            backend : The backend name or instance, the current one by
                      default.
        """
        backend = get_backend(backend)
        offsets, widths = self.routing["offsets"], self.port_widths
        index = backend.index(tuple(
            k for port in ports
            for k in range(offsets[port], offsets[port] + widths[port])),
            state)
        return backend.put(state, index, values.reshape(state.shape[0], -1))

    def forward(self, x=None, init=None, n_rounds: int = None,
                inject: bool = True, return_rounds: bool = False,
                return_flat: bool = False, memory=None,
                return_memory: bool = False, causal: bool = False,
                backend: str | Backend = None, modules=None):
        """
        Synchronous message passing along the wires of the map, i.e. the
        execution formula of the geometry of interaction, as the
        :class:`~discopy.neural.execution.Execution` of the map on a
        backend: all the messages in one flat array, the boxes sharing a
        module evaluated in one batched call per round and the routing one
        permutation.

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
                          collecting the per-box outputs.
            memory : The initial private memory, per box occurrence or as
                     one tensor of the concatenated memory dimensions.
            return_memory : Whether to return the final per-box memories
                            together with the usual result.
            causal : Whether to activate every box once in topological
                     order, for a feed-forward map; not combined with
                     ``n_rounds``.
            backend : The backend name or instance, the current one by
                      default.
            modules : The backend-owned modules in :attr:`modules` order,
                      the modules of the boxes by default.
        """
        if not (len(self.dom) or len(self.cod)) and x is not None\
                and x.shape[-1]:
            raise ValueError("A closed map takes no input.")
        execution = Execution(
            self, x, init, memory=memory, backend=backend, modules=modules)
        if causal:
            if n_rounds is not None:
                raise ValueError(
                    "A causal schedule cannot be combined with n_rounds.")
            if return_rounds or return_flat:
                raise ValueError(
                    "A causal schedule has no rounds to return.")
            return execution.forward_causal(inject, return_memory)
        return execution.forward(
            n_rounds, inject, return_memory, return_rounds, return_flat)

    __call__ = forward


Id = Diagram.id

Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
