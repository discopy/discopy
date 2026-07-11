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

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        sudoku_peers
        sudoku
        solve_sudoku
        random_sudoku
        check_sudoku
        decode_sudoku

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

import random
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

    def forward(self, x: "torch.Tensor" = None, init=None,
                n_rounds: int = None, inject: bool = True):
        """
        Synchronous message passing along the wires of the map, i.e. the
        execution formula of the geometry of interaction.

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

        given = [x] + (list(init)
                       if isinstance(init, (list, tuple)) else [init])
        ref = next((t for t in given if t is not None), None)
        batch_size = 1 if ref is None else ref.shape[0]
        proto = ref if ref is not None else next(
            iter(self.parameters()), None) if self.boxes else None

        def zeros(width):
            kwargs = {} if proto is None else {
                "dtype": proto.dtype, "device": proto.device}
            return torch.zeros(batch_size, width, **kwargs)

        input_ports = [i for i, port in enumerate(ports)
                       if port.kind == PortKind.INPUT]
        x_slices = dict(zip(input_ports, torch.split(
            x, [widths[i] for i in input_ports], dim=-1))) if x is not None\
            else {i: zeros(widths[i]) for i in input_ports}
        if init is not None and isinstance(init, torch.Tensor):
            init = torch.split(init, list(widths), dim=-1)

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

    __call__ = forward


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap


def sudoku_peers(n: int = 2) -> tuple[tuple[int, ...], ...]:
    """
    The sorted peers of each cell of an ``n ** 2 x n ** 2`` sudoku grid,
    i.e. the cells sharing a row, column or ``n x n`` block with it.

    Parameters:
        n : The size of a block, i.e. 2 for a 4x4 grid, 3 for 9x9.

    Example
    -------
    >>> peers = sudoku_peers()
    >>> peers[0]
    (1, 2, 3, 4, 5, 8, 12)
    >>> set(map(len, peers)), set(map(len, sudoku_peers(3)))
    ({7}, {20})
    """
    size = n * n

    def block(cell):
        return (cell // size // n) * n + (cell % size) // n

    return tuple(
        tuple(sorted(
            other for other in range(size * size) if other != cell and (
                other // size == cell // size
                or other % size == cell % size
                or block(other) == block(cell))))
        for cell in range(size * size))


def sudoku(n: int = 2, dim: int = 2, network: Network = None) -> CMap:
    """
    A sudoku grid encoded as a closed combinatorial map, with one box per
    cell wired to each of its peers.

    Every cell is the same shared :class:`Network` so that the grid has one
    set of weights and does not depend on any given puzzle: the clues are
    injected as initial messages, see :meth:`CMap.forward`.

    Parameters:
        n : The size of a block, i.e. 2 for a 4x4 grid, 3 for 9x9.
        dim : The width of the messages between cells.
        network : The shared cell network,
                  ``Network('cell', Dim(0), Dim(dim) ** len(peers))``
                  by default, with no module.

    Example
    -------
    >>> grid = sudoku()
    >>> len(grid.boxes), grid.n_ports
    (16, 112)
    >>> assert grid.edges.is_fixpoint_free_involution()
    >>> assert len(grid.connected_components) == 1
    >>> assert not grid.is_planar
    """
    peers = sudoku_peers(n)
    network = network if network is not None\
        else Network('cell', Dim(0), Dim(dim) ** len(peers[0]))
    wires = [
        ((cell, peers[cell].index(other)), (other, peers[other].index(cell)))
        for cell in range(n ** 4) for other in peers[cell] if cell < other]
    return CMap.from_wiring(n ** 4 * (network, ), wires)


def solve_sudoku(grid: tuple[int, ...], n: int = 2,
                 digits: tuple[int, ...] = None) -> tuple[int, ...] | None:
    """
    Solve a sudoku grid by backtracking, with 0 for blank cells.

    Parameters:
        grid : The ``n ** 4`` cells of the grid, row-major.
        digits : The order in which to try digits, increasing by default.

    Returns:
        The completed grid, or ``None`` when there is no solution.

    Example
    -------
    >>> solution = solve_sudoku((1, 0, 0, 0) + (0, ) * 12)
    >>> solution[:4]
    (1, 2, 3, 4)
    >>> assert check_sudoku(solution)
    >>> assert solve_sudoku((1, 1) + (0, ) * 14) is None
    """
    size = n * n
    peers = sudoku_peers(n)
    digits = tuple(range(1, size + 1)) if digits is None else digits
    grid = list(grid)

    def fill(cell):
        if cell == size * size:
            return True
        if grid[cell]:
            return all(grid[peer] != grid[cell] for peer in peers[cell])\
                and fill(cell + 1)
        for digit in digits:
            if all(grid[peer] != digit for peer in peers[cell]):
                grid[cell] = digit
                if fill(cell + 1):
                    return True
        grid[cell] = 0
        return False

    return tuple(grid) if fill(0) else None


def random_sudoku(n: int = 2, n_clues: int = 8, seed: int = 0
                  ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    A random sudoku puzzle and its solution, generated by backtracking with
    a shuffled digit order and then blanking random cells.

    Parameters:
        n : The size of a block, i.e. 2 for a 4x4 grid, 3 for 9x9.
        n_clues : The number of cells left as clues.
        seed : The seed of the random generator.

    Example
    -------
    >>> clues, solution = random_sudoku(seed=42)
    >>> assert check_sudoku(solution)
    >>> assert sum(clue != 0 for clue in clues) == 8
    >>> assert all(clue in (0, digit)
    ...            for clue, digit in zip(clues, solution))
    """
    size = n * n
    generator = random.Random(seed)
    digits = list(range(1, size + 1))
    generator.shuffle(digits)
    solution = solve_sudoku(size * size * (0, ), n, digits=tuple(digits))
    cells = list(range(size * size))
    generator.shuffle(cells)
    clues = list(solution)
    for cell in cells[:size * size - n_clues]:
        clues[cell] = 0
    return tuple(clues), solution


def check_sudoku(grid: tuple[int, ...], n: int = 2) -> bool:
    """
    Whether a completed grid satisfies all the sudoku constraints.

    Example
    -------
    >>> assert check_sudoku((
    ...     1, 2, 3, 4,
    ...     3, 4, 1, 2,
    ...     2, 1, 4, 3,
    ...     4, 3, 2, 1))
    >>> assert not check_sudoku((1, ) * 16)
    >>> assert not check_sudoku((0, ) * 16)
    """
    size = n * n
    peers = sudoku_peers(n)
    return all(digit in range(1, size + 1) for digit in grid) and all(
        grid[cell] != grid[peer]
        for cell in range(size * size) for peer in peers[cell])


def decode_sudoku(logits, clues: tuple[int, ...],
                  n: int = 2) -> tuple[int, ...]:
    """
    Decode per-cell logits into a completed grid, keeping the clues fixed
    and taking the most likely digit for each blank cell.

    Parameters:
        logits : One row of ``n ** 2`` scores per cell.
        clues : The cells of the puzzle, with 0 for blanks.

    Example
    -------
    >>> logits = 16 * [[.1, .2, .3, .4]]
    >>> decode_sudoku(logits, (1, 2) + (0, ) * 14)[:4]
    (1, 2, 4, 4)
    """
    del n
    result = []
    for cell, clue in enumerate(clues):
        if clue:
            result.append(clue)
        else:
            row = list(logits[cell])
            result.append(1 + max(range(len(row)), key=row.__getitem__))
    return tuple(result)
