# -*- coding: utf-8 -*-

"""
E-hypergraphs, i.e. string diagrams with alternatives as a table of cells.

An :class:`EHypergraph` is an open hypergraph with a union-find on its wires:
a table of cells, each an occurrence of a box whose row holds the wires on its
ports, together with a domain and a codomain. Rows are sharded by box and arity
so that each :class:`Shard` is a rectangular table of integers.

Each tree of the union-find is a vertex, i.e. the spider whose legs are its
member wires. Thus :meth:`EHypergraph.merge` fuses two vertices and a vertex
with more than one producing cell holds *alternatives*. This is the difference
with a :class:`discopy.hypergraph.Hypergraph`, which reads the same incidence
data as a Frobenius merge and cannot hold a formal sum at all.

E-hypergraphs are the arrows of a monoidal category: the tensor is their
disjoint union and composition merges the codomain of the first with the
domain of the second. They are equal when they have the same incidence data up
to renaming the wires, see :meth:`EHypergraph.setoid`, while
:meth:`EHypergraph.equiv` decides whether they are equal modulo the structure
of their category.

What that structure is, is read off the :class:`Supply` of the category, i.e.
which of the classes of :mod:`discopy.abc` it is an instance of. It decides
where each structural law holds:

* at lowering, where :meth:`EHypergraph.from_diagram` turns permutations,
  symmetric traces, and copy and discard when they are natural, into wiring,
* in the canonizers :meth:`EHypergraph.fuse_spiders` and
  :meth:`EHypergraph.yank_snakes`, which add the fused spider or the straight
  wire next to what they simplify, and
* in the congruence closure :meth:`EHypergraph.rebuild`, which merges the
  outputs of two occurrences of a box on the same inputs, unless the boxes of
  the category need not be functions.

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
    Supply
    EHypergraph

Example
-------
>>> from discopy.frobenius import Ty, Box, Swap, EHypergraph
>>> x, y = Ty('x'), Ty('y')
>>> f, g = Box('f', x, y), Box('g', y, x)
>>> print(EHypergraph.from_diagram(f >> g).to_diagram())
f >> g

The laws of the category hold on the nose, and so do the structural laws that
lowering turns into wiring:

>>> F = EHypergraph.from_diagram
>>> assert F(f) >> F(g) == F(f >> g)
>>> assert F(Swap(x, y) >> Swap(y, x)) == EHypergraph.id(x @ y)

Two occurrences of the same box on the same inputs are one row, and merging two
wires makes two rows congruent:

>>> graph = EHypergraph.id(x @ x)
>>> a, b = graph.dom_wires
>>> u, v = graph.intern(f, (a, )), graph.intern(f, (b, ))
>>> assert u != v and graph.intern(f, (a, )) == u
>>> graph.merge(a, b)
>>> graph.rebuild()
>>> assert graph.uf.find(u[0]) == graph.uf.find(v[0])
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Iterable, Iterator

import numpy as np

from discopy import hypergraph, markov, messages, rigid, symmetric, traced
from discopy.abc import (
    ColouredMonoid,
    HypergraphCategory,
    MarkovCategory,
    MonoidalCategory,
    NamedGeneric,
    RigidCategory,
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


@dataclass(frozen=True)
class Supply:
    """
    The structure a category supplies, which decides where its laws hold in
    an e-hypergraph: at lowering, in a canonizer or in the congruence.

    Parameters:
        symmetric : Whether permutations are wiring.
        trace : Whether symmetric traces are feedback, i.e. wiring.
        duals : Whether snakes are yanked, see
            :meth:`EHypergraph.yank_snakes`.
        frobenius : Whether spiders fuse, see
            :meth:`EHypergraph.fuse_spiders`.
        affine : Whether discarding is natural, i.e. wiring.
        cartesian : Whether copying is natural, i.e. wiring.

    Example
    -------
    >>> from discopy import frobenius, markov
    >>> Supply.of(markov.Diagram)  # doctest: +NORMALIZE_WHITESPACE
    Supply(symmetric=True, trace=True, duals=False, frobenius=False,
           affine=True, cartesian=False)
    >>> assert not Supply.of(markov.Diagram).hashcons
    >>> assert Supply.of(frobenius.Diagram).frobenius
    """
    symmetric: bool = False
    trace: bool = False
    duals: bool = False
    frobenius: bool = False
    affine: bool = False
    cartesian: bool = False

    @classmethod
    def of(cls, category: type) -> Supply:
        """
        The supply of a category, read off its classes in :mod:`discopy.abc`.

        A trace is feedback only when the category is symmetric: in a braided
        category, yanking a trace leaves a twist.

        Parameters:
            category : The category, e.g. :class:`discopy.markov.Diagram`.
        """
        is_symmetric = issubclass(category, SymmetricCategory)
        is_markov = issubclass(category, MarkovCategory)
        return cls(
            symmetric=is_symmetric,
            trace=is_symmetric and issubclass(category, TracedCategory),
            duals=issubclass(category, RigidCategory),
            frobenius=issubclass(category, HypergraphCategory),
            affine=is_markov and category.is_affine,
            cartesian=is_markov and category.is_cartesian)

    @property
    def discard(self) -> bool:
        """ Whether discarding is wiring. """
        return self.affine or self.cartesian

    @property
    def hashcons(self) -> bool:
        """
        Whether two occurrences of a box on the same inputs are congruent,
        i.e. whether boxes are functions: not in a Markov category that is not
        cartesian, where they are stochastic.
        """
        return self.cartesian or not self.affine


class EHypergraph(MonoidalCategory, NamedGeneric['category'], Setoid):
    """
    An open e-hypergraph, i.e. a table of cells over a union-find of wires
    with a domain and a codomain.

    An e-hypergraph is built in place by :meth:`wire`, :meth:`append`,
    :meth:`intern` and :meth:`merge`, and extended in place by the closure
    steps :meth:`rebuild`, :meth:`fuse_spiders` and :meth:`yank_snakes`,
    which only ever add cells and merge wires. Every other method, e.g. the
    composition and the tensor, returns a new e-hypergraph.

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
    >>> from discopy.frobenius import Ty, Box, EHypergraph
    >>> x = Ty('x')
    >>> f = Box('f', x, x)
    >>> graph = EHypergraph(
    ...     x, x, (f, ), ((0, ), (((0, ), (1, )), ), (1, )), [x, x])
    >>> assert graph == EHypergraph.from_diagram(f)
    >>> assert graph.shards[graph.boxes.intern(f), 1, 1][0] == (0, 1)
    """
    category = None
    ob = classproperty(lambda cls: cls.category.ob)
    supply = classproperty(lambda cls: Supply.of(cls.category))

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
        unless there is one already and boxes are functions, i.e. hash-consing.

        Parameters:
            box : The box to intern.
            src : The wires on its input ports.
        """
        key = (self.boxes.intern(box), len(box.dom), len(box.cod))
        gid = self.hashcons.get((key, self.signature(box, src)))
        if gid is not None and self.supply.hashcons:
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
        The vertices on the inputs of a cell, as a multiset for a spider.

        Parameters:
            box : The box of the cell.
            src : Its input wires.
        """
        vertices = tuple(map(self.uf.find, src))
        if self.supply.frobenius\
                and isinstance(box, self.category.spider_factory):
            return tuple(sorted(vertices))
        return vertices

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
        Close the e-hypergraph under congruence when boxes are functions, see
        :meth:`congruence`, and index the live cells for :meth:`intern`.
        """
        while self.supply.hashcons and self.congruence():
            pass
        self.hashcons = {
            (self.rows[gid][0], self.signature(box, src)): gid
            for gid, box, src, _ in self.scan()}

    def fuse_spiders(self) -> int:
        """
        Add a fused spider next to each connected set of spiders, see
        :meth:`fuse`, and merge both ends of each phaseless spider with one
        input and one output. Return the number of spiders added.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider, EHypergraph
        >>> x = Ty('x')
        >>> F = EHypergraph.from_diagram
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

    def yank_snakes(self) -> int:
        """
        Merge the two free ends of each snake, i.e. of a cup and a cap that
        share exactly one of their vertices, and return the number of snakes
        yanked.

        Example
        -------
        >>> from discopy.frobenius import Ty, Cup, Cap, EHypergraph
        >>> x = Ty('x')
        >>> snake = x @ Cap(x.r, x) >> Cup(x, x.r) @ x
        >>> graph = EHypergraph.from_diagram(snake)
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

    def saturate(self) -> EHypergraph:
        """
        The closure of the e-hypergraph under the canonizers its supply calls
        for and under congruence, applied in turn until neither changes it.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider, EHypergraph
        >>> x = Ty('x')
        >>> F = EHypergraph.from_diagram
        >>> graph = F(Spider(1, 2, x) >> Spider(2, 1, x))
        >>> assert graph.saturate() != graph
        """
        result, state = copy(self), None
        while state != (len(result.rows), len(set(result.uf))):
            state = (len(result.rows), len(set(result.uf)))
            if result.supply.frobenius:
                result.fuse_spiders()
            if result.supply.duals:
                result.yank_snakes()
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
            stable = True
            for gid, _, src, tgt in self.scan():
                if any(find(wire) not in cost for wire in src):
                    continue
                weight = 1 + sum(cost[find(wire)] for wire in src)
                for vertex in map(find, tgt):
                    if vertex not in cost or weight < cost[vertex]:
                        cost[vertex], chosen[vertex] = weight, gid
                        stable = False
            if stable:
                break
        for gid, _, _, tgt in self.scan():
            for vertex in map(find, tgt):
                chosen.setdefault(vertex, gid)
        return cost, chosen

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
    def from_diagram(cls, diagram: Diagram) -> EHypergraph:
        """
        Lower a diagram into a fresh e-hypergraph in one pass over its boxes,
        turning into wiring the structure the supply of its category allows.

        Parameters:
            diagram : The diagram to lower.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, EHypergraph
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> graph = EHypergraph.from_diagram(f >> f)
        >>> len(graph.rows), len(graph.uf)
        (2, 3)
        """
        factory = cls if cls.category else cls[type(diagram).factory]
        graph = factory(diagram.dom, diagram.cod)
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
        Lower one box onto its input wires and return its output wires:

        * a permutation permutes them when the category is symmetric,
        * a symmetric trace is feedback, see :meth:`lower_trace`,
        * a discard drops them and a copy repeats them when they are natural,
        * any other box is a cell.

        Parameters:
            box : The box to lower.
            src : The wires on its inputs.
        """
        supply = self.supply
        if supply.symmetric and isinstance(box, symmetric.Permutation):
            return [src[i] for i in box.perm]
        if supply.trace and isinstance(box, traced.Trace):
            return self.lower_trace(box, src)
        if supply.discard and isinstance(box, markov.Discard):
            return []
        if supply.cartesian and isinstance(box, markov.Copy):
            return src * len(box.cod)
        tgt = self.wires(box.cod)
        self.append(box, tuple(src), tgt)
        return list(tgt)

    def lower_trace(self, box: traced.Trace, src: list[int]) -> list[int]:
        """
        Lower a trace as feedback: lower its argument on fresh wires for the
        traced objects, then merge its traced outputs with them.

        When a traced output already is its input, the feedback would close a
        loop with nothing on it: the trace of the identity is a scalar cell.

        Parameters:
            box : The trace to lower.
            src : The wires on its inputs.
        """
        n_traced = len(box.arg.dom) - len(box.dom)
        typ = box.arg.dom[:n_traced] if box.left\
            else box.arg.dom[len(box.dom):]
        loops = list(self.wires(typ))
        tgt = self.lower(box.arg, loops + src if box.left else src + loops)
        feedback, tgt = (tgt[:n_traced], tgt[n_traced:]) if box.left\
            else (tgt[len(box.cod):], tgt[:len(box.cod)])
        for loop, back, obj in zip(loops, feedback, typ):
            if self.uf.find(loop) == self.uf.find(back):
                self.append(type(box)(box.arg.id(obj), left=box.left), (), ())
            else:
                self.merge(loop, back)
        return tgt

    @classmethod
    def id(cls, dom: Ty = None) -> EHypergraph:
        """
        The identity on a type, i.e. the same wires on both boundaries.

        Parameters:
            dom : The type.
        """
        dom = cls.ob() if dom is None else dom
        result = cls(dom, dom)
        result.dom_wires = result.cod_wires = result.wires(dom)
        return result

    @unbiased
    def tensor(self, other: EHypergraph) -> EHypergraph:
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
    def then(self, other: EHypergraph) -> EHypergraph:
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

    def equiv(self, other: EHypergraph) -> bool:
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
        >>> from discopy.frobenius import Ty, Spider, EHypergraph
        >>> x = Ty('x')
        >>> F = EHypergraph.from_diagram
        >>> assert F(Spider(1, 2, x) >> Spider(2, 1, x)).equiv(
        ...     EHypergraph.id(x))
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
        >>> from discopy.frobenius import Ty, Box, EHypergraph
        >>> x = Ty('x')
        >>> f, s = Box('f', x, x), Box('s', Ty(), x)
        >>> assert EHypergraph.from_diagram(f).to_hypergraph()\\
        ...     == f.to_hypergraph()
        >>> graph = EHypergraph.from_diagram(s @ s)
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
        >>> from discopy.frobenius import Ty, Box, EHypergraph
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g = Box('f', x, y), Box('g', x, y)
        >>> print(EHypergraph.from_diagram(f @ g).to_diagram())
        f @ g
        """
        return self.to_hypergraph().to_diagram()

    def __getitem__(self, gid: int) -> tuple[Box, tuple, tuple]:
        key, row = self.rows[gid]
        wires = self.shards[key][row]
        return self.boxes[key[0]], wires[:key[1]], wires[key[1]:]

    def __copy__(self) -> EHypergraph:
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
