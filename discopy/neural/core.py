# -*- coding: utf-8 -*-

"""
The compact closed category of bidirectional neural networks, with additive
dimensions as objects and concatenation as tensor.

A :class:`Network` with domain ``Dim(a_1, ..., a_m)`` and codomain
``Dim(b_1, ..., b_n)`` carries one module from ``R ** w`` to ``R ** w`` for
``w = a_1 + ... + a_m + b_1 + ... + b_n``, reading incoming messages on all
its ports and emitting outgoing messages on all its ports. Networks compose
with the cartesian product of vector spaces, so the tensor of dimensions is
their sum with the zero-dimensional space ``Dim(0)`` as unit; dimensions
are self-dual so that cups, caps and swaps are pure rerouting.

A morphism of this category is a graph neural network: the combinatorial
map of a diagram lays the messages of every port out as one flat vector,
:attr:`CMap.routing`, along which synchronous message passing computes the
execution formula of the geometry of interaction, see :cite:t:`Abramsky96`
and :mod:`discopy.interaction` for the Int-construction of Joyal, Street &
Verity :cite:p:`JoyalEtAl96`. Running it is the business of
:mod:`discopy.neural.execution`.

Cells need not be feedforward: a box can carry state between rounds along a
self-wired pair of ports.  Structurally that pair *is* the categorical trace
of the compact target -- it is wiring, which a functor preserves strictly --
while what it computes over finitely many rounds is delayed feedback: what a
box writes on one end it reads on the other one round later. Only the syntax
is compact closed: a trace would ask for the fixed point of the execution
formula, where a run computes a fixed number of rounds.

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

The combinatorial map of a snake has no box, it is pure rerouting:

>>> snake = Id(Dim(2)).transpose().to_map()
>>> snake.boxes, snake.routing["src"]
((), (2, 3, 0, 1))
"""

from __future__ import annotations

from functools import cached_property

from discopy import cmap, compact, hypergraph, monoidal, para
from discopy.cat import factory
from discopy.cmap import PortKind
from discopy.pivotal import Ty
from discopy.utils import assert_isinstance, factory_name, from_tree as decode


@factory
class Dim(monoidal.Dim, Ty):
    """
    A dimension is a tuple of positive integers seen as a self-dual type,
    with addition as tensor and the zero-dimensional space as unit.

    Example
    -------
    >>> assert Dim(0) == Dim() and Dim(0) @ Dim(2) @ Dim(3) == Dim(2, 3)
    >>> assert Dim(2, 3).l == Dim(2, 3).r == Dim(3, 2)
    >>> from discopy.utils import dumps, loads
    >>> assert loads(dumps(Dim(2, 3))) == Dim(2, 3)
    """
    neutral = 0
    l = r = property(lambda self: self.factory(*self.inside[::-1]))
    z = property(lambda self: 0)

    def unwind(self) -> "Dim":
        """ Dimensions are self-dual so their winding is trivial. """
        return self


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

    In the language of :mod:`discopy.para`, the module of a network
    :math:`f : X \\to Y` is a parametric map on the *boundary* of its box,

    .. math:: \\Phi_f : \\partial f \\otimes P_f \\to \\partial f, \\qquad
              \\partial f = X^* \\otimes Y,

    with :math:`P_f` the weights: it answers every leg of the box, inputs
    included, which no ordinary parametric map :math:`X \\otimes P \\to Y`
    -- a layer, composing by substitution as :class:`Para` does -- can say.
    Two interactions glued along a shared object do not compose by
    substitution either: they talk to each other along the wires, by
    symmetric feedback, i.e. the trace of the two boxes over the shared
    boundary, and what computes it is a finite number of rounds of
    :meth:`CMap.forward`; see :mod:`discopy.neural.map` for the global
    transition they add up to.

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
    The module is the ``data`` of the box, so networks compare equal when
    they have the same name, shape, memory and module, and a framework's
    modules compare by identity. The dagger and rotation of a network reuse
    its module and preserve its memory, with the public ports read in the
    new order: the dagger computes :math:`\\Phi_f` on the boundary read as
    ``cod @ dom``, which is the same interaction with its two halves
    exchanged only when the module is equivariant under that block swap.
    The repr and the serialisation omit the module, which has no
    eval-able representation, so ``eval(repr(f)) == f`` and
    ``loads(dumps(f)) == f`` hold for a network without one and give the
    shape of one with.

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
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        mem = f", mem={self.mem!r}" if self.mem else ""
        z = f", z={self.z}" if self.z else ""
        return f"{factory_name(type(self))}({self.name!r}, {self.dom!r}, "\
            f"{self.cod!r}{mem}{z})"

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
        return super().setoid() + (self.mem, )

    def to_tree(self) -> dict:
        """ Serialise the shape of the network, memory included. """
        tree = super().to_tree()
        tree.pop('data', None)
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
    >>> assert linear(2) == Para.generator("linear2", Dim(2), Dim(2), Dim(4))
    """
    category = Diagram

    @classmethod
    def generator(cls, name: str, dom: Dim, cod: Dim, param: Dim = Dim()
                  ) -> Para:
        """
        The parametric network of one generator, i.e. its box
        ``dom @ param -> cod`` with ``param`` as parameter object.

        Parameters:
            name : The name of the generator.
            dom : The domain of the generator.
            cod : The codomain of the generator.
            param : The parameter object, the unit by default.
        """
        return cls(dom, cod, Network(name, dom @ param, cod), param)


Equation = compact.Equation


class CMap(cmap.CMap[Diagram]):
    """
    A neural combinatorial map is a compact map with networks as boxes,
    which computes as a graph neural network: one message per port,
    travelling along the wires given by the ``edges`` involution, laid out
    as one flat vector by :attr:`routing`. Running it is the business of
    :mod:`discopy.neural.execution`.

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

    @cached_property
    def port_widths(self) -> tuple[int, ...]:
        """ The dimension carried by each port of the map. """
        return tuple(sum(port.obj.inside) for port in self.ports)

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

    @cached_property
    def memory_widths(self) -> tuple[int, ...]:
        """ The private memory width of each box occurrence. """
        return tuple(sum(box.mem.inside) for box in self.boxes)

    @cached_property
    def input_ports(self) -> tuple[int, ...]:
        """ The indices of the boundary input ports. """
        return tuple(i for i, port in enumerate(self.ports)
                     if port.kind == PortKind.INPUT)

    @cached_property
    def output_ports(self) -> tuple[int, ...]:
        """ The indices of the boundary output ports. """
        return tuple(i for i, port in enumerate(self.ports)
                     if port.kind == PortKind.OUTPUT)

    @cached_property
    def has_boundary(self) -> bool:
        """ Whether the map has any boundary port. """
        return bool(len(self.dom) or len(self.cod))

    @cached_property
    def routing(self) -> dict:
        """
        The wiring of the map as flat positions, with no tensor framework:

        * ``total`` : the total width, and ``offsets`` : the flat offset of
          each port,
        * ``src`` : the routing permutation, ``incoming = outgoing[src]``,
        * ``input``, ``output`` : the flat positions of the boundary ports,
          and ``ports`` : those of the ports of every box, in box order
          then logical order,
        * ``boxes`` : each box on its own, with its ``module`` index, its
          public ``width`` and ``memory_width``, the flat positions of its
          ``ports`` and of its private ``memory``, and the ``targets`` its
          outputs arrive at, i.e. the far end of each of its wires; and
          ``widths`` : the public width of each box,
        * ``groups`` : the boxes grouped by module, port widths and memory
          width, each with its ``ports`` and ``memory`` positions in box
          order, so that one module call evaluates a whole group at once.

        Example
        -------
        >>> f = Network('f', Dim(0), Dim(1, 1), module=object())
        >>> ring = CMap.from_wiring(
        ...     (f, f), [((0, 0), (1, 1)), ((0, 1), (1, 0))])
        >>> ring.routing["src"], ring.routing["ports"]
        ((3, 2, 1, 0), (1, 0, 3, 2))
        >>> ring.routing["boxes"][0]["targets"]
        (2, 3)
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

        box_ports = tuple(self.box_ports(i) for i in range(len(self.boxes)))
        boxes = tuple({
            "module": self.module_indices[i], "boxes": (i, ),
            "width": sum(widths[port] for port in ports),
            "memory_width": memory_widths[i], "ports": flat(ports),
            "memory": tuple(range(memory_offsets[i], memory_offsets[i + 1])),
            "targets": flat(tuple(self.edges[port] for port in ports))}
            for i, ports in enumerate(box_ports))
        groups: dict = {}
        for i, ports in enumerate(box_ports):
            key = (self.module_indices[i],
                   tuple(widths[port] for port in ports), memory_widths[i])
            groups.setdefault(key, []).append(i)
        return {
            "total": total, "offsets": tuple(offsets),
            "src": flat(tuple(self.edges)),
            "input": flat(self.input_ports), "output": flat(self.output_ports),
            "ports": tuple(k for box in boxes for k in box["ports"]),
            "boxes": boxes, "widths": tuple(box["width"] for box in boxes),
            "groups": tuple({
                "module": module, "boxes": tuple(members),
                "width": sum(box_widths), "memory_width": memory_width,
                "ports": tuple(k for i in members for k in boxes[i]["ports"]),
                "memory": tuple(
                    k for i in members for k in boxes[i]["memory"])}
                for (module, box_widths, memory_width), members
                in groups.items())}


Id = Diagram.id

Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
