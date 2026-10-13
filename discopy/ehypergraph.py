# -*- coding: utf-8 -*-

"""
E-hypergraphs, i.e. string diagrams with alternatives as a table of cells.

An e-hypergraph is an open hypergraph with a union-find on its wires: a table
of cells, each an occurrence of a box whose row holds the wires on its ports,
together with a domain and a codomain. Rows are sharded by box and arity so
that each :class:`Shard` is a rectangular table of integers.

Each tree of the union-find is a vertex, i.e. the spider whose legs are its
member wires. Thus :meth:`Symmetric.merge` fuses two vertices and a vertex
with more than one producing cell holds *alternatives*. This is the difference
with a :class:`discopy.hypergraph.Hypergraph`, which reads the same incidence
data as a Frobenius merge and cannot hold a formal sum at all.

E-hypergraphs over a symmetric `category` are the arrows of a symmetric
category: the tensor is their disjoint union, composition merges the codomain
of the first with the domain of the second and the swap is wiring. They are
equal when they have the same incidence data up to renaming the wires, see
:meth:`Symmetric.setoid`, while :meth:`Symmetric.equiv` decides whether they
are equal modulo the structure of their category.

As in :mod:`discopy.para`, each subclass assumes more structure of its
underlying `category` and specialises the methods where that structure holds:
:meth:`Symmetric.lower_box`, which turns it into wiring,
:meth:`Symmetric.canonize`, which adds what it simplifies next to what it
simplifies, and :meth:`Symmetric.congruence`, which merges the outputs of two
occurrences of a box on the same inputs.

* :class:`Symmetric` turns permutations into wiring,
* :class:`Traced` turns traces into feedback,
* :class:`Markov` keeps copy and discard as cells and two occurrences of a box
  apart, since its boxes need not be deterministic,
* :class:`Cartesian` assumes they are, turning copy and discard into wiring,
* :class:`Compact` yanks snakes, see :meth:`Compact.yank_snakes`, and
* :class:`Frobenius` fuses spiders, see :meth:`Frobenius.fuse_spiders`.

This is the enrichment of the ``metatheory`` equality saturation engine,
without its rewrite rules.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    SymbolTable
    Shard
    Symmetric
    Traced
    Markov
    Cartesian
    Compact
    Frobenius

Example
-------
>>> from discopy.symmetric import Ty, Box, Swap
>>> x, y = Ty('x'), Ty('y')
>>> f, g = Box('f', x, y), Box('g', y, x)
>>> print(Symmetric.from_diagram(f >> g).to_diagram())
f >> g

The laws of the category hold on the nose, and so do the structural laws that
lowering turns into wiring:

>>> F = Symmetric.from_diagram
>>> assert F(f) >> F(g) == F(f >> g)
>>> assert F(Swap(x, y) >> Swap(y, x)) == Symmetric.id(x @ y)

Two occurrences of the same box on the same inputs are one row, and merging two
wires makes two rows congruent:

>>> graph = Symmetric.id(x @ x)
>>> a, b = graph.dom_wires
>>> u, v = graph.intern(f, (a, )), graph.intern(f, (b, ))
>>> assert u != v and graph.intern(f, (a, )) == u
>>> graph.merge(a, b)
>>> graph.rebuild()
>>> assert graph.uf.find(u[0]) == graph.uf.find(v[0])
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Hashable, Iterable, Iterator

import numpy as np

from discopy import (
    compact, frobenius, hypergraph, markov, messages, rigid, symmetric, traced)
from discopy.abc import (
    ColouredMonoid,
    CompactCategory,
    HypergraphCategory,
    MarkovCategory,
    NamedGeneric,
    SymmetricCategory,
    TracedCategory,
)
from discopy.utils import (
    AxiomError,
    Setoid,
    UnionFind,
    assert_iscomposable,
    classproperty,
    factory_name,
    unbiased,
)

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


class Shard(ColouredMonoid):
    """
    The rows of an e-hypergraph that share a box and an arity.

    Shards form a monoid coloured by their arity ``(n_in, n_out)``, with the
    concatenation of their rows as tensor. Rows are also added one at a time
    with :meth:`append`, which grows the columns in place.

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
    >>> shard @ Shard(1, 2, [(6, 7, 8)]).shift(1)
    ehypergraph.Shard(1, 2, [(0, 1, 2), (3, 4, 5), (7, 8, 9)])
    """
    ob = tuple

    def __init__(self, n_in: int, n_out: int,
                 rows: list[tuple[int, ...]] = ()):
        self.n_in, self.n_out, self.length = n_in, n_out, 0
        self.columns = np.zeros((1, n_in + n_out), dtype=np.int64)
        for row in rows:
            self.append(row)

    @property
    def dom(self) -> tuple[int, int]:
        """ The colour of the shard, i.e. its arity. """
        return self.n_in, self.n_out

    cod = dom

    @classmethod
    def id(cls, dom: tuple[int, int]) -> Shard:
        """
        The empty shard of a given arity.

        Parameters:
            dom : The arity of the shard.
        """
        return cls(*dom)

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

    def tensor(self, *others: Shard) -> Shard:
        """
        The concatenation of the rows of shards with the same arity.

        Parameters:
            others : The other shards.
        """
        for other in others:
            assert_iscomposable(self, other)
        result = type(self)(self.n_in, self.n_out)
        result.columns = np.concatenate(
            [shard.columns[:shard.length] for shard in (self, ) + others])
        result.length = len(result.columns)
        return result

    def shift(self, offset: int) -> Shard:
        """
        The same rows with every wire shifted by an offset, e.g. to make room
        for the wires of another e-hypergraph.

        Parameters:
            offset : The number to add to each wire.
        """
        result = type(self)(self.n_in, self.n_out)
        result.columns = self.columns[:self.length] + offset
        result.length = self.length
        return result

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


class Symmetric(SymmetricCategory, NamedGeneric['category'], Setoid):
    """
    An open e-hypergraph over a symmetric `category`, i.e. a table of cells
    over a union-find of wires with a domain and a codomain, where the
    permutations of the category are wiring and two occurrences of a box on
    the same inputs are congruent.

    An e-hypergraph is built in place by :meth:`wire`, :meth:`append`,
    :meth:`intern` and :meth:`merge`, and extended in place by the closure
    steps :meth:`rebuild` and :meth:`canonize`, which only ever add cells and
    merge wires. Every other method, e.g. the composition and the tensor,
    returns a new e-hypergraph.

    Parameters:
        dom : The domain of the e-hypergraph.
        cod : The codomain of the e-hypergraph.
        boxes : The box of each cell.
        wires : The wires of the domain, the input and output wires of each
            cell, and the wires of the codomain.
        wire_types : The type of each wire.
        merges : Pairs of wires in the same vertex.

    Example
    -------
    >>> from discopy.symmetric import Ty, Box
    >>> x = Ty('x')
    >>> f = Box('f', x, x)
    >>> graph = Symmetric(
    ...     x, x, (f, ), ((0, ), (((0, ), (1, )), ), (1, )), [x, x])
    >>> assert graph == Symmetric.from_diagram(f)
    >>> assert graph.shards[graph.boxes.intern(f), 1, 1][0] == (0, 1)
    """
    category = symmetric.Diagram
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(self, dom: Ty, cod: Ty, boxes: tuple[Box, ...] = (),
                 wires: tuple = ((), (), ()), wire_types: list[Ty] = (),
                 merges: list[tuple[int, int]] = ()):
        self.dom, self.cod = dom, cod
        self.boxes, self.uf, self.wire_types = SymbolTable(), UnionFind(), []
        self.shards, self.hashcons, self.rows, self.dead = {}, {}, [], set()
        for typ in wire_types:
            self.wire(typ)
        dom_wires, box_wires, cod_wires = wires
        self.dom_wires, self.cod_wires = tuple(dom_wires), tuple(cod_wires)
        for box, (src, tgt) in zip(boxes, box_wires):
            self.append(box, tuple(src), tuple(tgt))
        for left, right in merges:
            self.merge(left, right)

    def check(self, wires: tuple[int, ...], typ: Ty):
        """
        Raise :class:`AxiomError` if some wires do not carry a given type.

        Parameters:
            wires : The wires to check.
            typ : The type they should carry.
        """
        types = [self.wire_types[wire] for wire in wires]
        if types != list(typ):
            raise AxiomError(messages.TYPE_ERROR.format(typ, types))

    def wire(self, typ: Ty) -> int:
        """
        Add a wire of a given atomic type and return it.

        Parameters:
            typ : The type of the wire.
        """
        self.wire_types.append(typ)
        return self.uf.fresh()

    def wires(self, typ: Ty) -> tuple[int, ...]:
        """
        Add one wire for each object of a type and return them.

        Parameters:
            typ : The type of the wires.
        """
        return tuple(self.wire(obj) for obj in typ)

    def append(self, box: Box, src: tuple[int, ...],
               tgt: tuple[int, ...]) -> int:
        """
        Add a cell with given input and output wires, and return its row.

        Parameters:
            box : The box of the cell.
            src : The wires on the input ports.
            tgt : The wires on the output ports.
        """
        self.check(src, box.dom)
        self.check(tgt, box.cod)
        key = (self.boxes.intern(box), len(box.dom), len(box.cod))
        shard = self.shards.setdefault(key, Shard(key[1], key[2]))
        shard.append(tuple(src) + tuple(tgt))
        self.rows.append((key, len(shard) - 1))
        self.hashcons.setdefault(
            (key, self.signature(box, src)), len(self.rows) - 1)
        return len(self.rows) - 1

    def intern(self, box: Box, src: tuple[int, ...]) -> tuple[int, ...]:
        """
        The output wires of a box on given input wires, adding a cell for it
        unless there is one already, i.e. hash-consing.

        Parameters:
            box : The box to intern.
            src : The wires on its input ports.
        """
        key = (self.boxes.intern(box), len(box.dom), len(box.cod))
        gid = self.hashcons.get((key, self.signature(box, src)))
        if gid is not None:
            return self[gid][2]
        tgt = self.wires(box.cod)
        self.append(box, src, tgt)
        return tgt

    def merge(self, left: int, right: int):
        """
        Assert that two wires are equal, i.e. fuse their vertices.

        Parameters:
            left : The first wire.
            right : The second wire.
        """
        self.check((right, ), self.wire_types[left])
        self.uf.union(left, right)

    def signature(self, box: Box, src: tuple[int, ...]) -> tuple[int, ...]:
        """
        The vertices on the inputs of a cell, i.e. what makes two occurrences
        of a box congruent.

        Parameters:
            box : The box of the cell.
            src : Its input wires.
        """
        # pylint: disable=unused-argument  # a spider reads it, see Frobenius
        return tuple(map(self.uf.find, src))

    def scan(self) -> Iterator[tuple[int, Box, tuple, tuple]]:
        """ The live cells in order, as a row with its box and its wires. """
        return ((gid, ) + self[gid] for gid in range(len(self.rows))
                if gid not in self.dead)

    def congruence(self) -> bool:
        """
        Merge the outputs of every live cell with those of the first live cell
        with the same box and signature, marking it dead, and return whether
        anything was merged.
        """
        index, merged = {}, False
        for gid, box, src, tgt in self.scan():
            first = index.setdefault((box, self.signature(box, src)), gid)
            if first == gid:
                continue
            for left, right in zip(self[first][2], tgt):
                self.merge(left, right)
            self.dead.add(gid)
            merged = True
        return merged

    def rebuild(self):
        """
        Close the e-hypergraph under congruence, see :meth:`congruence`, and
        index the live cells for :meth:`intern`.
        """
        while self.congruence():
            pass
        self.hashcons = {
            (self.rows[gid][0], self.signature(box, src)): gid
            for gid, box, src, _ in self.scan()}

    def canonize(self):
        """
        Add what the structure of the category simplifies, next to what it
        simplifies: nothing for a symmetric category, where the structure is
        all wiring already.
        """

    def saturate(self) -> Symmetric:
        """
        The closure of the e-hypergraph under :meth:`canonize` and
        :meth:`rebuild`, applied in turn until neither changes it.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider
        >>> x = Ty('x')
        >>> graph = Frobenius.from_diagram(Spider(1, 2, x) >> Spider(2, 1, x))
        >>> assert graph.saturate() != graph
        """
        result, state = copy(self), None
        while state != (len(result.rows), len(set(result.uf))):
            state = (len(result.rows), len(set(result.uf)))
            result.canonize()
            result.rebuild()
        return result

    def costs(self) -> tuple[dict[int, int], dict[int, int]]:
        """
        The least number of cells needed to produce each vertex, and the
        cheapest cell producing it, breaking ties by lowest row.
        """
        find = self.uf.find
        produced = {
            vertex for _, _, _, tgt in self.scan() for vertex in map(
                find, tgt)}
        cost = {vertex: 0 for vertex in map(find, range(len(self.uf)))
                if vertex not in produced}
        chosen = {}
        for _ in range(len(self.rows) + 1):
            if not self.relax(cost, chosen):
                break
        for gid, _, _, tgt in self.scan():
            for vertex in map(find, tgt):
                chosen.setdefault(vertex, gid)
        return cost, chosen

    def relax(self, cost: dict[int, int], chosen: dict[int, int]) -> bool:
        """
        One pass of the relaxation behind :meth:`costs`: each cell whose inputs
        all have a cost lowers the cost of its outputs, in place.

        Parameters:
            cost : The least known number of cells producing each vertex.
            chosen : The cheapest known cell producing each vertex.

        Returns whether any cost was lowered.
        """
        find, lowered = self.uf.find, False
        for gid, _, src, tgt in self.scan():
            if any(find(wire) not in cost for wire in src):
                continue
            weight = 1 + sum(cost[find(wire)] for wire in src)
            for vertex in map(find, tgt):
                if vertex not in cost or weight < cost[vertex]:
                    cost[vertex], chosen[vertex] = weight, gid
                    lowered = True
        return lowered

    def section(self, boundary: tuple[int, ...]) -> list[int]:
        """
        The cheapest cells that produce a given boundary, i.e. one alternative
        for each vertex, together with the cells that produce nothing.

        Parameters:
            boundary : The wires the section has to produce.
        """
        chosen, keep, scan = self.costs()[1], set(), list(boundary)
        for gid, _, src, tgt in self.scan():
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

    @classmethod
    def from_diagram(cls, diagram: Diagram) -> Symmetric:
        """
        Lower a diagram into a fresh e-hypergraph in one pass over its boxes,
        turning into wiring the structure of the category, see
        :meth:`lower_box`.

        Parameters:
            diagram : The diagram to lower.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> graph = Symmetric.from_diagram(f >> f)
        >>> len(graph.rows), len(graph.uf)
        (2, 3)
        """
        graph = cls(diagram.dom, diagram.cod)
        graph.dom_wires = graph.wires(diagram.dom)
        graph.cod_wires = tuple(graph.lower(diagram, list(graph.dom_wires)))
        return graph

    def lower(self, diagram: Diagram, scan: list[int]) -> list[int]:
        """
        Lower the boxes of a diagram onto a list of open wires, and return the
        open wires once they are all there.

        Parameters:
            diagram : The diagram to lower.
            scan : The wires on its domain.
        """
        for box, offset in zip(diagram.boxes, diagram.offsets):
            end = offset + len(box.dom)
            scan[offset:end] = self.lower_box(box, scan[offset:end])
        return scan

    def lower_box(self, box: Box, src: list[int]) -> list[int]:
        """
        Lower one box onto its input wires and return its output wires: a
        permutation permutes them, any other box is a cell.

        Parameters:
            box : The box to lower.
            src : The wires on its inputs.
        """
        if isinstance(box, symmetric.Permutation):
            return [src[i] for i in box.perm]
        tgt = self.wires(box.cod)
        self.append(box, tuple(src), tgt)
        return list(tgt)

    @classmethod
    def id(cls, dom: Ty = None) -> Symmetric:
        """
        The identity on a type, i.e. the same wires on both boundaries.

        Parameters:
            dom : The type.
        """
        dom = cls.ob() if dom is None else dom
        result = cls(dom, dom)
        result.dom_wires = result.cod_wires = result.wires(dom)
        return result

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Symmetric:
        """
        The swap of the underlying category, i.e. the same wires on both
        boundaries in a different order.

        Parameters:
            left : The type on the left of the swap.
            right : The type on the right of the swap.
        """
        return cls.from_diagram(cls.category.swap(left, right))

    @unbiased
    def tensor(self, other: Symmetric) -> Symmetric:
        """
        The tensor of two e-hypergraphs, i.e. their disjoint union.

        Parameters:
            other : The other e-hypergraph.
        """
        result, offset, shards = copy(self), len(self.uf), {}
        for typ in other.wire_types:
            result.wire(typ)
        for wire, label in enumerate(other.uf):
            if wire != label:
                result.uf.union(offset + wire, offset + label)
        for key, shard in other.shards.items():
            new = (result.boxes.intern(other.boxes[key[0]]), ) + key[1:]
            base = result.shards.get(new, Shard.id(key[1:]))
            shards[key] = new, len(base)
            result.shards[new] = base @ shard.shift(offset)
        for gid, (key, row) in enumerate(other.rows):
            result.rows.append((shards[key][0], shards[key][1] + row))
            if gid in other.dead:
                result.dead.add(len(result.rows) - 1)
        for gid in range(len(self.rows), len(result.rows)):
            box, src, _ = result[gid]
            result.hashcons.setdefault(
                (result.rows[gid][0], result.signature(box, src)), gid)
        result.dom, result.cod = self.dom @ other.dom, self.cod @ other.cod
        result.dom_wires += tuple(offset + w for w in other.dom_wires)
        result.cod_wires += tuple(offset + w for w in other.cod_wires)
        return result

    @unbiased
    def then(self, other: Symmetric) -> Symmetric:
        """
        The composition of two e-hypergraphs, i.e. their disjoint union with
        the codomain of the first merged with the domain of the second.

        Parameters:
            other : The other e-hypergraph.
        """
        assert_iscomposable(self, other)
        result, n, m = self.tensor(other), len(self.dom), len(self.cod)
        for left, right in zip(result.cod_wires[:m], result.dom_wires[n:]):
            result.merge(left, right)
        result.dom, result.cod = self.dom, other.cod
        result.dom_wires = result.dom_wires[:n]
        result.cod_wires = result.cod_wires[m:]
        return result

    def equiv(self, other: Symmetric) -> bool:
        """
        Whether two parallel e-hypergraphs are equal modulo the structure of
        their category, i.e. their codomains are the same vertices once their
        domains are merged and the result is saturated.

        This is not transitive: an e-hypergraph that holds two alternatives
        is equivalent to each of them, which need not be equivalent. It is
        also blind to scalars, i.e. to the cells that neither boundary depends
        on, since it only compares the codomains.

        Parameters:
            other : The other e-hypergraph.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider
        >>> x = Ty('x')
        >>> F = Frobenius.from_diagram
        >>> assert F(Spider(1, 2, x) >> Spider(2, 1, x)).equiv(
        ...     Frobenius.id(x))
        """
        if (self.dom, self.cod) != (other.dom, other.cod):
            return False
        union, n, m = self @ other, len(self.dom), len(self.cod)
        for left, right in zip(union.dom_wires[:n], union.dom_wires[n:]):
            union.merge(left, right)
        union = union.saturate()
        return all(union.uf.find(left) == union.uf.find(right) for left, right
                   in zip(union.cod_wires[:m], union.cod_wires[m:]))

    def incidence(self) -> hypergraph.Hypergraph:
        """
        The hypergraph with one box for each live cell and one spider for each
        vertex, i.e. the incidence data read as a Frobenius merge.
        """
        cells, labels = list(self.scan()), {}
        label = lambda wire: labels.setdefault(
            self.uf.find(wire), len(labels))
        dom_wires = tuple(map(label, self.dom_wires))
        box_wires = tuple((tuple(map(label, src)), tuple(map(label, tgt)))
                          for _, _, src, tgt in cells)
        cod_wires = tuple(map(label, self.cod_wires))
        spider_types = tuple(self.wire_types[root] for root in labels)
        factory = hypergraph.Hypergraph[type(self).category]
        return factory(self.dom, self.cod, tuple(box for _, box, *_ in cells),
                       (dom_wires, box_wires, cod_wires), spider_types)

    def setoid(self) -> hypergraph.Hypergraph:
        """
        The incidence data, so that two e-hypergraphs are equal when they have
        the same live cells on the same vertices up to renaming the wires.
        """
        return self.incidence()

    def to_hypergraph(self) -> hypergraph.Hypergraph:
        """
        The hypergraph of the cheapest section of the e-hypergraph producing
        its codomain, i.e. one alternative for each vertex.

        A cell with no inputs is copied once for each of the ports that
        consume it, so that hash-consing two occurrences of the same state
        does not turn them into one.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x = Ty('x')
        >>> f, s = Box('f', x, x), Box('s', Ty(), x)
        >>> assert Symmetric.from_diagram(f).to_hypergraph()\\
        ...     == f.to_hypergraph()
        >>> graph = Symmetric.from_diagram(s @ s)
        >>> assert len(graph.to_hypergraph().boxes) == 2
        """
        labels, spider_types = {}, {}

        def label(wire):
            root = self.uf.find(wire)
            if root not in labels:
                labels[root] = len(spider_types)
                spider_types[labels[root]] = self.wire_types[root]
            return labels[root]

        section = self.section(self.cod_wires)
        dom_wires = tuple(map(label, self.dom_wires))
        boxes = [self[gid][0] for gid in section]
        box_wires = [(tuple(map(label, self[gid][1])),
                      tuple(map(label, self[gid][2]))) for gid in section]
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
        cod_wires = tuple(map(consume, map(label, self.cod_wires)))
        factory = hypergraph.Hypergraph[type(self).category]
        return factory(self.dom, self.cod, tuple(boxes),
                       (dom_wires, tuple(box_wires), cod_wires), spider_types)

    def to_diagram(self) -> Diagram:
        """
        The diagram of the cheapest section of the e-hypergraph producing its
        codomain, see :meth:`to_hypergraph`.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', x, y)
        >>> print(Symmetric.from_diagram(f @ g).to_diagram())
        f @ g
        """
        return self.to_hypergraph().to_diagram()

    def __getitem__(self, gid: int) -> tuple[Box, tuple, tuple]:
        key, row = self.rows[gid]
        wires = self.shards[key][row]
        return self.boxes[key[0]], wires[:key[1]], wires[key[1]:]

    def __copy__(self) -> Symmetric:
        result = type(self)(self.dom, self.cod)
        result.boxes = SymbolTable(self.boxes.inside)
        result.uf.parent = list(self.uf.parent)
        result.uf.size = list(self.uf.size)
        result.wire_types, result.rows = list(self.wire_types), list(self.rows)
        result.shards = {
            key: shard.shift(0) for key, shard in self.shards.items()}
        result.hashcons, result.dead = dict(self.hashcons), set(self.dead)
        result.dom_wires, result.cod_wires = self.dom_wires, self.cod_wires
        return result

    def __repr__(self) -> str:
        cells = list(self.scan())
        boxes = tuple(box for _, box, *_ in cells)
        wires = (self.dom_wires, tuple(
            (src, tgt) for *_, src, tgt in cells), self.cod_wires)
        merges = [(wire, label) for wire, label in enumerate(self.uf)
                  if wire != label]
        return factory_name(type(self)) + f"(dom={self.dom!r}, "\
            f"cod={self.cod!r}, boxes={boxes!r}, wires={wires!r}, "\
            f"wire_types={self.wire_types!r}, merges={merges!r})"


class Traced(Symmetric, TracedCategory):
    """
    An e-hypergraph over a traced symmetric `category`, where the trace is
    feedback, i.e. wiring.

    Example
    -------
    >>> from discopy.symmetric import Ty, Swap, Id
    >>> x = Ty('x')
    >>> assert Traced.from_diagram(Swap(x, x).trace())\\
    ...     == Traced.from_diagram(Id(x)) == Traced.swap(x, x).trace()
    """
    def trace(self, n: int = 1, left: bool = False) -> Traced:
        """
        The trace of an e-hypergraph, i.e. its traced outputs merged with its
        traced inputs, see :meth:`feedback`.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """
        result, dom, cod = copy(self), len(self.dom) - n, len(self.cod) - n
        if left:
            loops, backs = result.dom_wires[:n], result.cod_wires[:n]
            result.dom, result.cod = self.dom[n:], self.cod[n:]
            result.dom_wires = result.dom_wires[n:]
            result.cod_wires = result.cod_wires[n:]
        else:
            loops, backs = result.dom_wires[dom:], result.cod_wires[cod:]
            result.dom, result.cod = self.dom[:dom], self.cod[:cod]
            result.dom_wires = result.dom_wires[:dom]
            result.cod_wires = result.cod_wires[:cod]
        result.feedback(loops, backs, left)
        return result

    def feedback(self, loops: tuple[int, ...], backs: tuple[int, ...],
                 left: bool = False):
        """
        Merge each traced output with its traced input, in place.

        When a traced output already is its input, the feedback would close a
        loop with nothing on it: the trace of the identity is a scalar cell.

        Parameters:
            loops : The traced inputs.
            backs : The traced outputs.
            left : Whether the trace is on the left.
        """
        for loop, back in zip(loops, backs):
            if self.uf.find(loop) == self.uf.find(back):
                arg = self.category.id(self.wire_types[loop])
                self.append(self.category.trace_factory(arg, left), (), ())
            else:
                self.merge(loop, back)

    def lower_box(self, box: Box, src: list[int]) -> list[int]:
        """
        Lower one box onto its input wires, a trace as feedback, see
        :meth:`lower_trace`.

        Parameters:
            box : The box to lower.
            src : The wires on its inputs.
        """
        if isinstance(box, traced.Trace):
            return self.lower_trace(box, src)
        return super().lower_box(box, src)

    def lower_trace(self, box: traced.Trace, src: list[int]) -> list[int]:
        """
        Lower a trace as feedback: lower its argument on fresh wires for the
        traced objects, then merge its traced outputs with them.

        Parameters:
            box : The trace to lower.
            src : The wires on its inputs.
        """
        n_traced = len(box.arg.dom) - len(box.dom)
        typ = box.arg.dom[:n_traced] if box.left\
            else box.arg.dom[len(box.dom):]
        loops = list(self.wires(typ))
        tgt = self.lower(box.arg, loops + src if box.left else src + loops)
        backs, tgt = (tgt[:n_traced], tgt[n_traced:]) if box.left\
            else (tgt[len(box.cod):], tgt[:len(box.cod)])
        self.feedback(loops, backs, box.left)
        return tgt


class Markov(Symmetric, MarkovCategory):
    """
    An e-hypergraph over a Markov `category`, i.e. a copy-discard category,
    where copy and discard are cells and the boxes need not be deterministic:
    two occurrences of a box on the same inputs are two cells.

    Example
    -------
    >>> from discopy.markov import Ty, Box
    >>> x = Ty('x')
    >>> graph = Markov.id(x)
    >>> a, = graph.dom_wires
    >>> f = Box('f', x, x)
    >>> assert graph.intern(f, (a, )) != graph.intern(f, (a, ))
    """
    category = markov.Diagram

    @classmethod
    def copy(cls, x: Ty, n: int = 2) -> Markov:
        """
        The copy of the underlying category, as a cell.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.from_diagram(cls.category.copy(x, n))

    def intern(self, box: Box, src: tuple[int, ...]) -> tuple[int, ...]:
        """
        The output wires of a new cell for a box on given input wires, since
        two occurrences of a box need not be the same.

        Parameters:
            box : The box to intern.
            src : The wires on its input ports.
        """
        tgt = self.wires(box.cod)
        self.append(box, src, tgt)
        return tgt

    def congruence(self) -> bool:
        """
        Merge nothing, since two occurrences of a box on the same inputs need
        not be the same, and return that nothing was merged.
        """
        return False


class Cartesian(Markov):
    """
    An e-hypergraph over a Markov `category` whose boxes are deterministic and
    discard natural, i.e. a cartesian category, where copy and discard are
    wiring and two occurrences of a box on the same inputs are congruent.

    Example
    -------
    >>> from discopy.markov import Ty, Copy
    >>> x = Ty('x')
    >>> graph = Cartesian.from_diagram(Copy(x))
    >>> assert not graph.rows and len(set(graph.uf)) == 1
    """
    intern, congruence = Symmetric.intern, Symmetric.congruence

    def lower_box(self, box: Box, src: list[int]) -> list[int]:
        """
        Lower one box onto its input wires: a discard drops them, a copy
        repeats them.

        Parameters:
            box : The box to lower.
            src : The wires on its inputs.
        """
        if isinstance(box, markov.Discard):
            return []
        if isinstance(box, markov.Copy):
            return src * len(box.cod)
        return super().lower_box(box, src)


class Compact(Traced, CompactCategory):
    """
    An e-hypergraph over a compact `category`, where cups and caps are cells
    and :meth:`canonize` yanks the snakes they make, see :meth:`yank_snakes`.
    """
    category = compact.Diagram

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> Compact:
        """
        The cups of the underlying category, as cells.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """
        return cls.from_diagram(cls.category.cups(left, right))

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> Compact:
        """
        The caps of the underlying category, as cells.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """
        return cls.from_diagram(cls.category.caps(left, right))

    def canonize(self):
        """ Yank the snakes, see :meth:`yank_snakes`. """
        super().canonize()
        self.yank_snakes()

    def yank_snakes(self) -> int:
        """
        Merge the two free ends of each snake, i.e. of a cup and a cap that
        share exactly one of their vertices, and return the number of snakes
        yanked.

        Example
        -------
        >>> from discopy.frobenius import Ty, Cup, Cap
        >>> x = Ty('x')
        >>> snake = x @ Cap(x.r, x) >> Cup(x, x.r) @ x
        >>> graph = Compact.from_diagram(snake)
        >>> graph.yank_snakes()
        1
        >>> assert graph.uf.find(graph.dom_wires[0])\\
        ...     == graph.uf.find(graph.cod_wires[0])
        """
        find, caps = self.uf.find, {}
        for _, box, _, tgt in self.scan():
            if isinstance(box, rigid.Cap):
                for wire in tgt:
                    caps.setdefault(find(wire), []).append(tgt)
        cups = [src for _, box, src, _ in self.scan()
                if isinstance(box, rigid.Cup)]
        candidates = lambda cup: dict.fromkeys(
            cap for wire in cup for cap in caps.get(find(wire), ()))
        return sum(
            self.yank(cup, cap) for cup in cups for cap in candidates(cup))

    def yank(self, cup: tuple[int, int], cap: tuple[int, int]) -> bool:
        """
        Merge the free ends of a cup and a cap when they make a snake, and
        return whether they did.

        Parameters:
            cup : The input wires of the cup.
            cap : The output wires of the cap.
        """
        find = self.uf.find
        shared = set(map(find, cup)) & set(map(find, cap))
        if len(shared) != 1 or len(set(map(find, cup))) != 2\
                or len(set(map(find, cap))) != 2:
            return False
        left, = (wire for wire in cup if find(wire) not in shared)
        right, = (wire for wire in cap if find(wire) not in shared)
        if self.wire_types[left] != self.wire_types[right]:
            return False
        self.merge(left, right)
        return True


class Frobenius(Compact, Markov, HypergraphCategory):
    """
    An e-hypergraph over a hypergraph `category`, where spiders are cells,
    :meth:`canonize` fuses them, see :meth:`fuse_spiders`, and two
    occurrences of a box on the same inputs are congruent as in
    :class:`Symmetric`, the inputs of a spider being a multiset.

    Example
    -------
    >>> from discopy.frobenius import Ty, Box
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y), Box('g', y, x)
    >>> print(Frobenius.from_diagram(f >> g).to_diagram())
    f >> g
    """
    category = frobenius.Diagram
    intern, congruence = Symmetric.intern, Symmetric.congruence

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Ty) -> Frobenius:
        """
        The spiders of the underlying category, as cells.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return cls.from_diagram(
            cls.category.spiders(n_legs_in, n_legs_out, typ))

    def signature(self, box: Box, src: tuple[int, ...]) -> tuple[int, ...]:
        """
        The vertices on the inputs of a cell, as a multiset for a spider.

        Parameters:
            box : The box of the cell.
            src : Its input wires.
        """
        vertices = super().signature(box, src)
        if isinstance(box, self.category.spider_factory):
            return tuple(sorted(vertices))
        return vertices

    def canonize(self):
        """ Fuse the spiders, then yank the snakes. """
        self.fuse_spiders()
        super().canonize()

    def fuse_spiders(self) -> int:
        """
        Add a fused spider next to each connected set of spiders, see
        :meth:`fuse`, and merge both ends of each phaseless spider with one
        input and one output. Return the number of spiders added.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider
        >>> x = Ty('x')
        >>> F = Frobenius.from_diagram
        >>> graph = F(Spider(1, 2, x) >> Spider(2, 1, x))
        >>> graph.fuse_spiders()
        1
        >>> assert graph.uf.find(graph.dom_wires[0])\\
        ...     == graph.uf.find(graph.cod_wires[0])
        """
        fused = sum(map(self.fuse, self.spider_components()))
        for _, box, src, tgt in list(self.scan()):
            if isinstance(box, self.category.spider_factory)\
                    and len(src) == len(tgt) == 1 and not box.phase:
                self.merge(src[0], tgt[0])
        return fused

    def spider_components(self) -> list[list[tuple]]:
        """
        The connected sets of at least two live spiders of the same kind and
        type, where two spiders are connected when one produces a vertex the
        other consumes, leaving out the phaseless spiders whose input already
        is their output.
        """
        find, kind = self.uf.find, lambda box: (type(box), box.typ)
        spiders = [
            (gid, box, src, tgt) for gid, box, src, tgt in self.scan()
            if isinstance(box, self.category.spider_factory) and not (
                len(src) == len(tgt) == 1 and find(src[0]) == find(tgt[0]))]
        consumers, components = {}, UnionFind(range(len(spiders)))
        for i, (_, box, src, _) in enumerate(spiders):
            for wire in src:
                consumers.setdefault((kind(box), find(wire)), []).append(i)
        for i, (_, box, _, tgt) in enumerate(spiders):
            for wire in tgt:
                for j in consumers.get((kind(box), find(wire)), ()):
                    components.union(i, j)
        result = {}
        for i, label in enumerate(components):
            result.setdefault(label, []).append(spiders[i])
        return [component for component in result.values()
                if len(component) > 1]

    def fuse(self, component: list[tuple]) -> int:
        """
        Add the fused spider of a connected set of spiders, unless it has no
        legs, it is an identity or it is there already, and return the number
        of spiders added.

        A vertex is an input of the fused spider when a spider consumes it and
        either no spider produces it or something else does, i.e. another cell
        or the domain. Dually for the outputs. The phase of the fused spider
        is the sum of theirs.

        Parameters:
            component : The live spiders to fuse, as given by :meth:`scan`.
        """
        find, inside = self.uf.find, {gid for gid, *_ in component}
        others = [cell for cell in self.scan() if cell[0] not in inside]
        produced_outside = set(map(find, self.dom_wires)) | {
            find(wire) for *_, tgt in others for wire in tgt}
        consumed_outside = set(map(find, self.cod_wires)) | {
            find(wire) for _, _, src, _ in others for wire in src}
        inputs = [wire for _, _, src, _ in component for wire in src]
        outputs = [wire for *_, tgt in component for wire in tgt]
        produced, consumed = set(map(find, outputs)), set(map(find, inputs))
        legs_in = self.legs(
            wire for wire in inputs if find(wire) not in produced
            or find(wire) in produced_outside)
        legs_out = self.legs(
            wire for wire in outputs if find(wire) not in consumed
            or find(wire) in consumed_outside)
        phases = [box.phase for _, box, *_ in component if box.phase]
        spider = self.category.spider_factory(
            len(legs_in), len(legs_out), component[0][1].typ,
            sum(phases[1:], phases[0]) if phases else None)
        signature = (spider, tuple(map(find, legs_in)),
                     tuple(map(find, legs_out)))
        existing = {(box, tuple(map(find, src)), tuple(map(find, tgt)))
                    for _, box, src, tgt in self.scan()}
        identity = len(legs_in) == len(legs_out) == 1 and not phases\
            and find(legs_in[0]) == find(legs_out[0])
        if not legs_in + legs_out or identity or signature in existing:
            return 0
        self.append(spider, legs_in, legs_out)
        return 1

    def legs(self, wires: Iterable[int]) -> tuple[int, ...]:
        """
        One wire for each vertex, in order: two legs of a spider on the same
        vertex are one, by the special law.

        Parameters:
            wires : The wires on the legs.
        """
        result = {}
        for wire in wires:
            result.setdefault(self.uf.find(wire), wire)
        return tuple(result.values())
