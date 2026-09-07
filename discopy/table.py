# -*- coding: utf-8 -*-

"""
Diagrams as tables, i.e. the carrier of an e-graph of string diagrams.

A :class:`Carrier` is a table of cells over a union-find of wires. A cell is
one occurrence of a box: its row holds the wires on the box's input and
output ports. Rows are sharded by generator and arity so that each
:class:`Shard` is a rectangular table of integers.

Each tree of the union-find is a vertex of the underlying hypergraph, i.e. the
spider whose legs are its member wires. Thus :meth:`Carrier.merge` fuses two
vertices and a vertex with more than one producing cell holds *alternatives*.
This is what makes a carrier an e-graph rather than a
:class:`discopy.hypergraph.Hypergraph`, which reads the same incidence data as
a Frobenius merge and cannot hold a formal sum at all.

Composition is a side effect on the carrier: :meth:`Morphism.then` asserts that
the wires it composes are equal. Diagrams stay pure, the carrier is the
effectful codomain of :meth:`Carrier.from_diagram`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    SymbolTable
    UnionFind
    Shard
    Carrier
    Wires
    Morphism

Example
-------
>>> from discopy.frobenius import Ty, Box, Carrier
>>> x, y = Ty('x'), Ty('y')
>>> f, g = Box('f', x, y), Box('g', y, x)
>>> morphism = Carrier.from_diagram(f >> g)
>>> print(morphism.to_diagram())
f >> g

Two occurrences of the same box on the same wires are one row, and merging two
wires makes two rows congruent:

>>> carrier = Carrier()
>>> a, b = carrier.wires(x), carrier.wires(x)
>>> u, v = carrier.intern(f, a.inside), carrier.intern(f, b.inside)
>>> assert u != v and carrier.intern(f, a.inside) == u
>>> carrier.merge(a.inside[0], b.inside[0])
>>> carrier.rebuild()
>>> assert carrier.uf.find(u[0]) == carrier.uf.find(v[0])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Iterator

import numpy as np

from discopy import hypergraph, messages
from discopy.abc import MonoidalCategory, NamedGeneric
from discopy.utils import AxiomError, classproperty, factory_name, unbiased

if TYPE_CHECKING:
    from discopy.monoidal import Box, Diagram, Ty


def grow(column: np.ndarray, length: int) -> np.ndarray:
    """
    Copy a column into one at least twice as long, and at least ``length``.

    Parameters:
        column : The column to copy.
        length : The number of rows the result must hold.

    Example
    -------
    >>> grow(np.zeros(2, dtype=int), 3).shape
    (4,)
    """
    capacity = max(2 * len(column), length, 1)
    result = np.zeros((capacity, ) + column.shape[1:], dtype=column.dtype)
    result[:len(column)] = column
    return result


class SymbolTable:
    """
    A table from hashable symbols to consecutive rows.

    Symbols are keyed by type as well as value, so that ``1`` and ``True`` get
    two different rows.

    Parameters:
        inside : The symbols to intern, in order.

    Example
    -------
    >>> table = SymbolTable(["f"])
    >>> table.intern("g"), table.intern("f"), table.intern(1), table.intern(1.)
    (1, 0, 2, 3)
    >>> table[1], len(table)
    ('g', 4)
    """
    def __init__(self, inside: list[Hashable] = ()):
        self.inside, self.index = [], {}
        for symbol in inside:
            self.intern(symbol)

    def intern(self, symbol: Hashable) -> int:
        """
        The row of a symbol, appending it when it is not there yet.

        Parameters:
            symbol : The symbol to look up.
        """
        key = (type(symbol), symbol)
        if key not in self.index:
            self.index[key] = len(self.inside)
            self.inside.append(symbol)
        return self.index[key]

    def __getitem__(self, row: int) -> Hashable:
        return self.inside[row]

    def __len__(self) -> int:
        return len(self.inside)

    def __eq__(self, other) -> bool:
        return isinstance(other, SymbolTable) and self.inside == other.inside

    def __repr__(self) -> str:
        return f"{factory_name(type(self))}({self.inside})"


class UnionFind:
    """
    A union-find over wires, i.e. the vertices of a carrier.

    Each tree is one vertex of the underlying hypergraph, the spider whose legs
    are the wires in the tree. Roots are chosen by size, ties by lowest wire,
    so that the table does not depend on the order of the merges.

    Parameters:
        parent : The parent of each wire, the identity for a fresh one.

    Example
    -------
    >>> uf = UnionFind()
    >>> a, b, c = uf.fresh(), uf.fresh(), uf.fresh()
    >>> uf.union(b, c)
    >>> uf.find(b), uf.find(c), uf.find(a)
    (1, 1, 0)
    >>> uf
    table.UnionFind([0, 1, 1])
    """
    def __init__(self, parent: list[int] = ()):
        parent = list(parent)
        self.parent = np.array(parent + [0], dtype=np.int64)
        self.size = np.ones(len(self.parent), dtype=np.int64)
        self.length = len(parent)
        for wire, root in enumerate(parent):
            if wire != root:
                self.union(wire, root)

    def fresh(self) -> int:
        """ Add a wire in a tree of its own and return it. """
        if self.length == len(self.parent):
            self.parent = grow(self.parent, self.length + 1)
            self.size = grow(self.size, self.length + 1)
        self.parent[self.length] = self.length
        self.size[self.length] = 1
        self.length += 1
        return self.length - 1

    def find(self, wire: int) -> int:
        """
        The root of the tree containing a wire, compressing the path to it.

        Parameters:
            wire : The wire to look up.
        """
        root = wire
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[wire] != root:
            self.parent[wire], wire = root, self.parent[wire]
        return int(root)

    def union(self, left: int, right: int):
        """
        Merge the trees of two wires, i.e. fuse two vertices.

        Parameters:
            left : The first wire.
            right : The second wire.
        """
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if (self.size[left], -left) < (self.size[right], -right):
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def __len__(self) -> int:
        return self.length

    def __eq__(self, other) -> bool:
        return isinstance(other, UnionFind) and list(self) == list(other)

    def __iter__(self) -> Iterator[int]:
        return (self.find(wire) for wire in range(self.length))

    def __repr__(self) -> str:
        return f"{factory_name(type(self))}({list(self)})"


class Shard:
    """
    The rows of a carrier that share a generator and an arity.

    Parameters:
        n_in : The number of input ports of every row.
        n_out : The number of output ports of every row.
        rows : The wires of each row, inputs then outputs.

    Example
    -------
    >>> shard = Shard(1, 2, [(0, 1, 2)])
    >>> shard.append((3, 4, 5))
    >>> shard.src.tolist(), shard.tgt.tolist()
    ([[0], [3]], [[1, 2], [4, 5]])
    >>> shard
    table.Shard(1, 2, [(0, 1, 2), (3, 4, 5)])
    """
    def __init__(self, n_in: int, n_out: int,
                 rows: list[tuple[int, ...]] = ()):
        self.n_in, self.n_out, self.length = n_in, n_out, 0
        self.columns = np.zeros((1, n_in + n_out), dtype=np.int64)
        for row in rows:
            self.append(row)

    def append(self, row: tuple[int, ...]):
        """
        Add one row, growing the columns when they are full.

        Parameters:
            row : The wires of the row, inputs then outputs.
        """
        if len(row) != self.n_in + self.n_out:
            raise ValueError
        if self.length == len(self.columns):
            self.columns = grow(self.columns, self.length + 1)
        self.columns[self.length] = row
        self.length += 1

    @property
    def src(self) -> np.ndarray:
        """ The input columns of the rows. """
        return self.columns[:self.length, :self.n_in]

    @property
    def tgt(self) -> np.ndarray:
        """ The output columns of the rows. """
        return self.columns[:self.length, self.n_in:]

    def __getitem__(self, row: int) -> tuple[int, ...]:
        return tuple(int(wire) for wire in self.columns[row])

    def __len__(self) -> int:
        return self.length

    def __eq__(self, other) -> bool:
        return isinstance(other, Shard) and (
            self.n_in, self.n_out, list(self)) == (
                other.n_in, other.n_out, list(other))

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return (self[row] for row in range(self.length))

    def __repr__(self) -> str:
        return factory_name(type(self))\
            + f"({self.n_in}, {self.n_out}, {list(self)})"


class Carrier(NamedGeneric['category']):
    """
    A table of cells over a union-find of wires.

    Parameters:
        wire_types : The type of each wire, in order.
        cells : The box and the wires of each cell, inputs then outputs.
        merges : The pairs of wires to merge once the cells are interned.

    Example
    -------
    >>> from discopy.frobenius import Ty, Box, Carrier
    >>> x = Ty('x')
    >>> f = Box('f', x, x)
    >>> carrier = Carrier([x, x], [(f, 0, 1)])
    >>> assert carrier.cells == ((f, 0, 1), )
    >>> assert carrier.shards[carrier.boxes.intern(f), 1, 1][0] == (0, 1)
    """
    category = None
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(self, wire_types: list[Ty] = (),
                 cells: list[tuple] = (), merges: list[tuple] = ()):
        self.boxes, self.uf = SymbolTable(), UnionFind()
        self.shards, self.hashcons, self.rows = {}, {}, []
        self.dead, self.wire_types = set(), []
        for typ in wire_types:
            self.wire(typ)
        for box, *wires in cells:
            self.append(box, tuple(wires[:len(box.dom)]),
                        tuple(wires[len(box.dom):]))
        for left, right in merges:
            self.merge(left, right)

    def wire(self, typ: Ty) -> int:
        """
        Add a wire of a given atomic type and return it.

        Parameters:
            typ : The type of the wire.
        """
        self.wire_types.append(typ)
        return self.uf.fresh()

    def wires(self, typ: Ty) -> Wires:
        """
        Add one wire for each object of a type and return them.

        Parameters:
            typ : The type of the wires.
        """
        return Wires(self, tuple(self.wire(obj) for obj in typ), typ)

    def append(self, box: Box, src: tuple[int, ...],
               tgt: tuple[int, ...]) -> int:
        """
        Add a cell with given input and output wires, and return its row.

        Parameters:
            box : The box of the cell.
            src : The wires on the input ports.
            tgt : The wires on the output ports.
        """
        key = (self.boxes.intern(box), len(box.dom), len(box.cod))
        shard = self.shards.setdefault(key, Shard(key[1], key[2]))
        shard.append(tuple(src) + tuple(tgt))
        self.rows.append((key, len(shard) - 1))
        self.hashcons[key, tuple(map(self.uf.find, src))] = len(self.rows) - 1
        return len(self.rows) - 1

    def intern(self, box: Box, src: tuple[int, ...]) -> tuple[int, ...]:
        """
        The output wires of a box on given input wires, adding a cell for it
        when there is not one already, i.e. hash-consing.

        Parameters:
            box : The box to intern.
            src : The wires on its input ports.
        """
        key = (self.boxes.intern(box), len(box.dom), len(box.cod))
        canon = tuple(map(self.uf.find, src))
        if (key, canon) in self.hashcons:
            return self[self.hashcons[key, canon]][2]
        tgt = tuple(self.wire(obj) for obj in box.cod)
        self.append(box, tuple(src), tgt)
        return tgt

    def merge(self, left: int, right: int):
        """
        Assert that two wires are equal, i.e. fuse their vertices.

        Parameters:
            left : The first wire.
            right : The second wire.
        """
        self.uf.union(left, right)

    def rebuild(self):
        """
        Close the carrier under congruence: when two live cells have the same
        box on the same input classes, their output wires get merged and the
        later cell is marked dead.
        """
        while True:
            index, stable = {}, True
            for gid, box, src, tgt in self.scan():
                key = (box, tuple(map(self.uf.find, src)))
                if index.setdefault(key, gid) == gid:
                    continue
                for left, right in zip(self[index[key]][2], tgt):
                    self.uf.union(left, right)
                self.dead.add(gid)
                stable = False
            if stable:
                break
        self.hashcons = {
            (self.rows[gid][0], tuple(map(self.uf.find, src))): gid
            for gid, box, src, tgt in self.scan()}

    def scan(self) -> Iterator[tuple[int, Box, tuple, tuple]]:
        """ The live cells in order, as a row with its box and its wires. """
        return ((gid, ) + self[gid] for gid in range(len(self.rows))
                if gid not in self.dead)

    @property
    def cells(self) -> tuple[tuple, ...]:
        """ The box and the wires of each cell, live or dead. """
        return tuple((box, ) + src + tgt for box, src, tgt in map(
            self.__getitem__, range(len(self.rows))))

    def costs(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        The least number of cells needed to produce each vertex, and the
        cheapest cell producing it, breaking ties by lowest row.
        """
        produced = {
            wire for _, _, _, tgt in self.scan() for wire in map(
                self.uf.find, tgt)}
        cost = {wire: 0 for wire in set(self.uf) if wire not in produced}
        chosen = {}
        for _ in range(len(self.rows) + 1):
            stable = True
            for gid, box, src, tgt in self.scan():
                if any(self.uf.find(wire) not in cost for wire in src):
                    continue
                weight = 1 + sum(cost[self.uf.find(w)] for w in src)
                for wire in map(self.uf.find, tgt):
                    if wire not in cost or weight < cost[wire]:
                        cost[wire], chosen[wire] = weight, gid
                        stable = False
            if stable:
                break
        return cost, chosen

    def section(self, boundary: tuple[int, ...]) -> list[int]:
        """
        The cheapest cells that produce a given boundary, i.e. one alternative
        per vertex, together with the cells that produce nothing.

        Parameters:
            boundary : The wires the section has to produce.
        """
        chosen, keep, scan = self.costs()[1], set(), list(boundary)
        for gid, box, src, tgt in self.scan():
            if not tgt:
                keep.add(gid)
                scan += list(src)
        while scan:
            gid = chosen.get(self.uf.find(scan.pop()))
            if gid is None or gid in keep:
                continue
            keep.add(gid)
            scan += list(self[gid][1])
        return sorted(keep)

    def from_box(self, box: Box) -> Morphism:
        """
        The morphism of a box on fresh input wires.

        Parameters:
            box : The box to intern.
        """
        dom = self.wires(box.dom)
        cod = Wires(self, self.intern(box, dom.inside), box.cod)
        return Morphism(dom, cod)

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> Morphism:
        """
        Intern a diagram into a fresh carrier, one cell for each box.

        Parameters:
            diagram : The diagram to intern.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Carrier
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> carrier = Carrier.from_diagram(f >> f).carrier
        >>> len(carrier.rows), len(carrier.uf)
        (2, 3)
        """
        factory = cls if cls.category else cls[type(diagram).ar]
        carrier = factory()
        dom = carrier.wires(diagram.dom)
        scan = list(dom.inside)
        for box, offset in zip(diagram.boxes, diagram.offsets):
            scan[offset:offset + len(box.dom)] = carrier.intern(
                box, tuple(scan[offset:offset + len(box.dom)]))
        return Morphism(dom, Wires(carrier, tuple(scan), diagram.cod))

    def __getitem__(self, gid: int) -> tuple[Box, tuple, tuple]:
        key, row = self.rows[gid]
        wires = self.shards[key][row]
        return self.boxes[key[0]], wires[:key[1]], wires[key[1]:]

    def __eq__(self, other) -> bool:
        return isinstance(other, Carrier) and (
            self.category, self.wire_types, self.cells, self.uf) == (
                other.category, other.wire_types, other.cells, other.uf)

    def __repr__(self) -> str:
        merges = [(wire, root) for wire, root in enumerate(self.uf)
                  if wire != root]
        return factory_name(type(self))\
            + f"({self.wire_types}, {list(self.cells)}, {merges})"


@dataclass(frozen=True, eq=False)
class Wires:
    """
    The wires on the boundary of a :class:`Morphism`, i.e. an object of the
    category presented by a carrier.

    Parameters:
        carrier : The carrier holding the wires.
        inside : The wires themselves.
        ty : The type they carry.

    Example
    -------
    >>> from discopy.frobenius import Ty, Carrier
    >>> x, y = Ty('x'), Ty('y')
    >>> carrier = Carrier()
    >>> assert (carrier.wires(x) @ carrier.wires(y)).ty == x @ y
    """
    carrier: Carrier
    inside: tuple[int, ...]
    ty: Ty

    @unbiased
    def tensor(self, other: Wires) -> Wires:
        """
        Juxtapose the wires of two objects.

        Parameters:
            other : The other object.
        """
        if self.carrier is not other.carrier:
            raise AxiomError(messages.TYPE_ERROR.format(
                self.carrier, other.carrier))
        return Wires(
            self.carrier, self.inside + other.inside, self.ty @ other.ty)

    __matmul__ = tensor

    def __eq__(self, other) -> bool:
        return isinstance(other, Wires) and self.carrier is other.carrier\
            and (self.inside, self.ty) == (other.inside, other.ty)

    def __len__(self) -> int:
        return len(self.inside)


class Morphism(MonoidalCategory):
    """
    An arrow of the category presented by a carrier, i.e. a pair of boundaries.

    Composition asserts that the wires it composes are equal, so the laws of a
    monoidal category hold up to :meth:`equiv` rather than on the nose.

    Parameters:
        dom : The wires on the domain.
        cod : The wires on the codomain.

    Example
    -------
    >>> from discopy.frobenius import Ty, Box, Carrier
    >>> x = Ty('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> carrier = Carrier()
    >>> u, v = map(carrier.from_box, (f, g))
    >>> assert (u >> v).dom == u.dom and (u >> v).cod == v.cod
    >>> assert Morphism.id(u.dom).then(u).equiv(u)
    """
    ob = Wires

    def __init__(self, dom: Wires, cod: Wires):
        self.dom, self.cod = dom, cod

    @property
    def carrier(self) -> Carrier:
        """ The carrier this morphism is a boundary of. """
        return self.dom.carrier

    @classmethod
    def id(cls, dom: Wires) -> Morphism:
        """
        The identity on an object, i.e. the same wires twice.

        Parameters:
            dom : The object.
        """
        return cls(dom, dom)

    def is_composable(self, other: Morphism) -> bool:
        return self.carrier is other.carrier and self.cod.ty == other.dom.ty

    @unbiased
    def then(self, other: Morphism) -> Morphism:
        """
        Compose two morphisms by merging the wires on their shared boundary.

        Parameters:
            other : The other morphism.
        """
        if not self.is_composable(other):
            raise AxiomError(messages.NOT_COMPOSABLE.format(
                self, other, self.cod.ty, other.dom.ty))
        for left, right in zip(self.cod.inside, other.dom.inside):
            self.carrier.merge(left, right)
        return Morphism(self.dom, other.cod)

    @unbiased
    def tensor(self, other: Morphism) -> Morphism:
        """
        Juxtapose two morphisms.

        Parameters:
            other : The other morphism.
        """
        return Morphism(self.dom @ other.dom, self.cod @ other.cod)

    def equiv(self, other: Morphism) -> bool:
        """
        Whether two morphisms have the same boundary up to the equations
        asserted in the carrier.

        Parameters:
            other : The other morphism.
        """
        find = self.carrier.uf.find
        return self.carrier is other.carrier and all(
            len(x) == len(y) and all(map(
                lambda i, j: find(i) == find(j), x.inside, y.inside))
            for x, y in [(self.dom, other.dom), (self.cod, other.cod)])

    def to_hypergraph(self) -> hypergraph.Hypergraph:
        """
        The hypergraph of the cheapest section of the carrier producing this
        morphism, i.e. one alternative for each vertex.

        A cell with no inputs is copied once for each of the ports that
        consume it, so that hash-consing two occurrences of the same state
        does not turn them into one.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Carrier
        >>> x = Ty('x')
        >>> f, s = Box('f', x, x), Box('s', Ty(), x)
        >>> assert Carrier.from_diagram(f).to_hypergraph()\\
        ...     == f.to_hypergraph()
        >>> assert len(Carrier.from_diagram(s @ s).to_hypergraph().boxes) == 2
        """
        carrier, labels, spider_types = self.carrier, {}, {}

        def label(wire):
            root = carrier.uf.find(wire)
            if root not in labels:
                labels[root] = len(spider_types)
                spider_types[labels[root]] = carrier.wire_types[root]
            return labels[root]

        section = carrier.section(self.cod.inside)
        dom_wires = tuple(map(label, self.dom.inside))
        boxes = [carrier[gid][0] for gid in section]
        box_wires = [(tuple(map(label, carrier[gid][1])),
                      tuple(map(label, carrier[gid][2]))) for gid in section]
        states = {cod[0]: i for i, (dom, cod) in enumerate(box_wires)
                  if not dom and len(cod) == 1}
        occupied = set()

        def consume(spider):
            if spider not in occupied or spider not in states:
                occupied.add(spider)
                return spider
            spider_types[len(spider_types)] = spider_types[spider]
            boxes.append(boxes[states[spider]])
            box_wires.append(((), (len(spider_types) - 1, )))
            return len(spider_types) - 1

        for i in range(len(section)):
            box_wires[i] = (
                tuple(map(consume, box_wires[i][0])), box_wires[i][1])
        cod_wires = tuple(map(consume, map(label, self.cod.inside)))
        factory = hypergraph.Hypergraph[type(carrier).category]
        return factory(self.dom.ty, self.cod.ty, tuple(boxes),
                       (dom_wires, tuple(box_wires), cod_wires), spider_types)

    def to_diagram(self) -> Diagram:
        """
        The diagram of the cheapest section of the carrier producing this
        morphism, see :meth:`to_hypergraph`.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Carrier
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', x, y)
        >>> print(Carrier.from_diagram(f @ g).to_diagram())
        f @ g
        """
        return self.to_hypergraph().to_diagram()

    def __eq__(self, other) -> bool:
        return isinstance(other, Morphism)\
            and (self.dom, self.cod) == (other.dom, other.cod)

    def __repr__(self) -> str:
        return f"{factory_name(type(self))}({self.dom}, {self.cod})"
