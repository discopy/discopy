# -*- coding: utf-8 -*-

"""
The free hypergraph category with cospans of labeled hypergraphs as arrows.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Spider
    Wires
    Boundary
    Wiring
    SpiderTypes
    Hypergraph

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        pushout
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property
from inspect import isclass
from itertools import chain

import random
from typing import Any, Iterable, Union, TYPE_CHECKING

import matplotlib.pyplot as plt

from networkx import (
    DiGraph as Graph,
    spring_layout,
    draw_networkx,
    has_path,
    dag_longest_path_length,
    topological_sort,
    weisfeiler_lehman_graph_hash,
)
from networkx.algorithms.isomorphism import is_isomorphic

from discopy import cmap, messages
from discopy.abc import (
    HypergraphCategory, MarkovCategory, MonoidalCategory, NamedGeneric)
from discopy.drawing import Node
from discopy.python.finset import Permutation
from discopy.utils import (
    factory_name,
    assert_isinstance,
    pushout,
    unbiased,
    AxiomError,
    assert_isatomic,
    assert_istraceable,
    classproperty,
    tuplify,
    untuplify,
)
if TYPE_CHECKING:
    from discopy.monoidal import Ty, Box, Diagram


Spider = Any
""" The labels of spiders can be of any type. """

Wires = tuple[Spider, ...]
""" Wires are n-tuples of :class:`Spider` labels. """

Boundary = tuple[Wires, Wires]
""" A boundary is a pair of input and output :class:`Wires`. """

Wiring = tuple[Wires, tuple[Boundary, ...], Wires]
"""
The wiring of a hypergraph is given by a triple
``dom_wires, box_wires, cod_wires`` where
``(dom_wires, cod_wires)`` is the :class:`Boundary` of the overall hypergraph
while ``box_wires`` are the boundaries for each of its boxes.
"""

SpiderTypes = Union[Mapping[Spider, "Ty"], Iterable["Ty"]]
"""
Mapping from :class:`Spider` to atomic :class:`frobenius.Ty`.
"""


class Hypergraph(MonoidalCategory, NamedGeneric['category']):
    """
    A hypergraph is given by:

    - a domain, a codomain and an n-tuple of boxes
    - a :class:`Wiring` triple ``dom_wires, box_wires, cod_wires`` where
        - ``(dom_wires, cod_wires)`` is the :class:`Boundary` of the hypergraph
        - ``box_wires: tuple[Boundary, ...]`` is the boundary of each box
    - an optional mapping :class:`SpiderTypes` from spiders to types

    A :class:`Boundary` is just a pair of input and output :class:`Wires`.

    :class:`Wires` are n-tuples of :class:`Spider` labels.

    :class:`Spider` labels can be of any type.

    Parameters:
        dom (category.ob) : The domain of the diagram, i.e. its input.
        cod (category.ob) :
            The codomain of the diagram, i.e. its output.
        boxes (tuple[category, ...]) : The boxes inside the diagram.
        wires (Wiring) : List of wires from ports to spiders.
        spider_types (SpiderTypes) :
            Optional mapping from spiders to atomic types, if ``None`` then
            this is computed from the types of ports.
        offsets : tuple[int | None, ...]
            Number of wires left of each box, used by :meth:`to_diagram`.

    Note
    ----
    Abstractly, a hypergraph diagram can be seen as a cospan for the boundary::

        range(len(dom)) -> range(n_spiders) <- range(len(cod))

    together with a cospan for each box in boxes::

        range(len(box.dom)) -> range(n_spiders) <- range(len(box.cod))

    Composition of two hypergraph diagram is given by the pushout of the span::

        range(self.n_spiders) <- range(len(self.cod)) -> range(other.n_spiders)

    Note
    ----
    The ``Hypergraph`` class is parameterised by a ``Category``, i.e. instances
    of ``Hypergraph[C]`` have ``dom: C.ob`` and ``cod: C.ob`` as boundary and
    ``boxes: tuple[C.ar, ...]`` as generators. For example:

    >>> from discopy.frobenius import Hypergraph as H
    >>> from discopy.frobenius import Ty, Diagram
    >>> assert H.category.ob == Ty and H.category == Diagram

    The :class:`Functor` used by :meth:`from_diagram` is read off the category
    itself, i.e. ``H.functor == H.category.functor_factory``.

    >>> from discopy.frobenius import Functor
    >>> assert H.functor == Functor

    Thus subclasses of :class:`Diagram` get hypergraphs of their own category
    without a dedicated ``Hypergraph`` class, e.g. ``Hypergraph[Diagram]``.

    Examples
    --------
    >>> x, y, z = map(Ty, "xyz")

    >>> assert H.id(x @ y @ z).n_spiders == 3
    >>> assert H.id(x @ y @ z).wires ==((0, 1, 2), (), (0, 1, 2))

    >>> assert H.swap(x, y).n_spiders == 2
    >>> assert H.swap(x, y).wires == ((0, 1), (), (1, 0))

    >>> assert H.spiders(1, 2, x @ y).n_spiders == 2
    >>> assert H.spiders(1, 2, x @ y).wires ==((0, 1), (), (0, 1, 0, 1))
    >>> assert H.spiders(0, 0, x @ y @ z).n_spiders == 3
    >>> assert H.spiders(0, 0, x @ y @ z).wires == ((), (), ())

    >>> from discopy.frobenius import Box
    >>> f, g = Box('f', x, y).to_hypergraph(), Box('g', y, z).to_hypergraph()

    >>> assert f.n_spiders == g.n_spiders == 2
    >>> assert f.wires == g.wires == ((0, ), (((0, ), (1, )), ), (1, ))

    >>> assert (f >> g).n_spiders == 3
    >>> assert (f >> g).wires == ((0,), (((0,), (1,)), ((1,), (2,))), (2,))

    >>> assert (f @ g).n_spiders == 4
    >>> assert (f @ g).wires == ((0, 1), (((0,), (2,)), ((1,), (3,))), (2, 3))
    """
    category = None

    functor = classproperty(lambda cls: cls.category.functor_factory)
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(
            self, dom: Ty, cod: Ty, boxes: tuple[Box, ...],
            wires: Wiring, spider_types: SpiderTypes = None,
            offsets: tuple[int | None, ...] = None):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        self.dom, self.cod, self.boxes = dom, cod, boxes
        dom_wires, box_wires, cod_wires = wires

        if len(dom_wires) != len(dom):
            raise ValueError
        if len(cod_wires) != len(cod):
            raise ValueError
        if len(box_wires) != len(boxes):
            raise ValueError
        for box, (box_dom_wires, box_cod_wires) in zip(boxes, box_wires):
            if len(box_dom_wires) != len(box.dom):
                raise ValueError
            if len(box_cod_wires) != len(box.cod):
                raise ValueError

        flat_wires = dom_wires + tuple(
            chain.from_iterable(x + y for x, y in box_wires)) + cod_wires
        connected_spiders = set(flat_wires)

        if spider_types is None:
            spider_types = {spider: port.obj for spider, port in zip(
                flat_wires, self.ports)}
        if not isinstance(spider_types, Mapping):
            spider_types = dict(enumerate(spider_types))

        first_occurrence = {}
        for index, spider in enumerate(flat_wires):
            first_occurrence.setdefault(spider, index)
        relabeling = sorted(connected_spiders, key=first_occurrence.get)
        relabeling += sorted(set(spider_types.keys()) - connected_spiders)
        self.spider_types = tuple(
            spider_types[s].unwind() for s in relabeling)
        new_index = {s: i for i, s in enumerate(relabeling)}
        self.flat_wires = tuple(new_index[s] for s in flat_wires)
        self.wires = self.rebracket(self.flat_wires)
        self.dom_wires, self.box_wires, self.cod_wires = self.wires

        ports = self.ports
        for obj in self.spider_types:
            assert_isatomic(obj, self.category.ob)
        for obj, (producers, consumers) in zip(
                self.spider_types, self.spider_wires):
            for i in producers | consumers:
                if ports[i].obj.unwind() != obj:
                    raise AxiomError(messages.TYPE_ERROR.format(
                        obj, ports[i].obj))
            same_side = producers if len(producers) == 2 else\
                consumers if len(consumers) == 2 else None
            if same_side is not None:
                left, right = (ports[i].obj for i in sorted(same_side))
                if getattr(left, "r", left) != right\
                        and getattr(right, "r", right) != left:
                    raise AxiomError(messages.NOT_ADJOINT.format(left, right))

        self.offsets = offsets or tuple(len(boxes) * [None])

    @cached_property
    def spider_wires(self) -> list[tuple[set[int], set[int]]]:
        """
        The input and output wires for each spider of a hypergraph.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> for wires in (f >> H.spiders(1, 2, y)).spider_wires: print(wires)
        ({0}, {1})
        ({2}, {3, 4})
        """
        result = [(set(), set()) for _ in range(self.n_spiders)]
        for port, spider in enumerate(self.dom_wires):
            result[spider][0].add(port)
        n_ports = len(self.dom)
        for dom_wires, cod_wires in self.box_wires:
            for port, spider in enumerate(dom_wires):
                result[spider][1].add(port + n_ports)
            for port, spider in enumerate(cod_wires):
                result[spider][0].add(port + n_ports + len(dom_wires))
            n_ports += len(dom_wires + cod_wires)
        for port, spider in enumerate(self.cod_wires):
            result[spider][1].add(port + n_ports)
        return result

    @cached_property
    def ports(self):
        """
        The ports in a diagram.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H

        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ y).to_hypergraph()
        >>> g = Box('g', y @ y, z).to_hypergraph()
        >>> for port in (f >> g).ports: print(port)
        Node('input', i=0, obj=x)
        Node('dom', depth=0, i=0, obj=x)
        Node('cod', depth=0, i=0, obj=y)
        Node('cod', depth=0, i=1, obj=y)
        Node('dom', depth=1, i=0, obj=y)
        Node('dom', depth=1, i=1, obj=y)
        Node('cod', depth=1, i=0, obj=z)
        Node('output', i=0, obj=z)
        """
        inputs = [Node("input", i=i, obj=obj)
                  for i, obj in enumerate(self.dom)]
        doms_and_cods = [
            Node(kind, depth=depth, i=i, obj=obj)
            for depth, box in enumerate(self.boxes)
            for kind, typ in [("dom", box.dom), ("cod", box.cod)]
            for i, obj in enumerate(typ)]
        outputs = [Node("output", i=i, obj=obj)
                   for i, obj in enumerate(self.cod)]
        return inputs + doms_and_cods + outputs

    def rebracket(
            self, flat_wires: list[Spider], boxes=None, dom=None):
        """
        Rebracket a flat list of :class:`Spider` into a proper :class:`Wiring`.
        """
        dom = self.dom if dom is None else dom
        boxes = self.boxes if boxes is None else boxes
        box_wires, i = [], len(dom)
        dom_wires = tuple(flat_wires[:i])
        for depth, box in enumerate(boxes):
            box_wires.append(tuple(map(tuple, (
                flat_wires[i:i + len(box.dom)],
                flat_wires[i + len(box.dom):i + len(box.dom @ box.cod)]))))
            i += len(box.dom @ box.cod)
        cod_wires = tuple(flat_wires[i:])
        return (dom_wires, tuple(box_wires), cod_wires)

    @property
    def n_spiders(self):
        """ The number of spiders in a hypergraph diagram. """
        return len(self.spider_types)

    @property
    def scalar_spiders(self):
        """ The zero-legged spiders in a hypergraph diagram. """
        return [
            i for i, (x, y) in enumerate(self.spider_wires) if not x and not y]

    @classmethod
    def id(cls, dom=None) -> Hypergraph:
        dom = cls.category.ob() if dom is None else dom
        dom_wires = cod_wires = tuple(range(len(dom)))
        return cls(dom, dom, (), (dom_wires, (), cod_wires))

    twist = id

    @unbiased
    def then(self, other: Hypergraph):
        """
        Composition of two hypergraph diagrams, i.e. their :func:`pushout`.
        """
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod = self.dom, other.cod
        boxes, offsets = self.boxes + other.boxes, self.offsets + other.offsets
        left, right = pushout(
            self.n_spiders, other.n_spiders, self.cod_wires, other.dom_wires)
        relabel = lambda d, ls: tuple(d[i] for i in ls)
        dom_wires = relabel(left, self.dom_wires)
        box_wires = tuple(
            (relabel(left, x), relabel(left, y)) for x, y in self.box_wires)
        box_wires += tuple(
            (relabel(right, x), relabel(right, y)) for x, y in other.box_wires)
        cod_wires = relabel(right, other.cod_wires)
        wires = (dom_wires, box_wires, cod_wires)
        spider_types = {
            left[i]: t for i, t in enumerate(self.spider_types)}
        spider_types.update({
            right[i]: t for i, t in enumerate(other.spider_types)})
        return type(self)(dom, cod, boxes, wires, spider_types, offsets)

    @unbiased
    def tensor(self, other: Hypergraph):
        """ Tensor of two hypergraph diagrams, i.e. their disjoint union. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes, offsets = self.boxes + other.boxes, self.offsets + other.offsets
        shift = lambda w: tuple(self.n_spiders + i for i in w)
        dom_wires = self.dom_wires + shift(other.dom_wires)
        box_wires = self.box_wires + tuple(
            (shift(x), shift(y)) for x, y in other.box_wires)
        cod_wires = self.cod_wires + shift(other.cod_wires)
        wires = dom_wires, box_wires, cod_wires
        spiders = self.spider_types + other.spider_types
        return type(self)(dom, cod, boxes, wires, spiders, offsets)

    def dagger(self):
        """
        Dagger of a hypergraph diagram, called with :code:`[::-1]`.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> g = Box('g', y, z).to_hypergraph()
        >>> assert (f >> g)[::-1] == g[::-1] >> f[::-1]
        >>> assert H.spiders(1, 2, x @ y)[::-1] == H.spiders(2, 1, x @ y)
        """
        dom, cod = self.cod, self.dom
        boxes = tuple(box.dagger() for box in self.boxes[::-1])
        box_wires = tuple((y, x) for x, y in self.box_wires[::-1])
        wires = self.cod_wires, box_wires, self.dom_wires
        return type(self)(
            dom, cod, boxes, wires, self.spider_types, self.offsets[::-1])

    @classmethod
    def swap(cls, left, right):
        dom, cod, boxes = left @ right, right @ left, ()
        dom_wires = tuple(range(len(dom)))
        cod_wires = tuple(range(len(left), len(dom))) + tuple(range(len(left)))
        return cls(dom, cod, boxes, (dom_wires, (), cod_wires))

    braid = swap

    @classmethod
    def permutation(cls, xs: list[int], dom) -> Hypergraph:
        """
        The hypergraph that encodes a given permutation, with the same
        semantics as :meth:`discopy.symmetric.Diagram.permutation` but
        built directly as wires rather than as a chain of swap boxes,
        i.e. in time linear rather than quadratic in ``len(xs)``.

        Parameters:
            xs : A list of integers representing a permutation.
            dom : A type of the same length as ``xs``.
        """
        if list(range(len(dom))) != sorted(xs):
            raise ValueError(messages.WRONG_PERMUTATION.format(len(dom), xs))
        cod, boxes = dom.ob(*(dom.inside[i] for i in xs)), ()
        dom_wires, cod_wires = tuple(range(len(dom))), tuple(xs)
        return cls(dom, cod, boxes, (dom_wires, (), cod_wires))

    @classmethod
    def spiders(cls, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, spider_types = (), tuple(typ)
        dom_wires = n_legs_in * tuple(range(len(typ)))
        cod_wires = n_legs_out * tuple(range(len(typ)))
        return cls(dom, cod, boxes, (dom_wires, (), cod_wires), spider_types)

    @classmethod
    def copy(cls, typ, n=2) -> Hypergraph:
        return cls.spiders(1, n, typ)

    @classmethod
    def merge(cls, typ, n=2) -> Hypergraph:
        return cls.spiders(n, 1, typ)

    cup_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cup_factory(left, right)))
    cap_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cap_factory(left, right)))

    @classmethod
    def cups(cls, left, right):
        adjoint = left.r if hasattr(left, "r") else left[::-1]
        if adjoint != right:
            raise AxiomError
        dom_wires = tuple(range(len(left))) + tuple(reversed(range(len(left))))
        return cls(
            left @ right, cls.category.ob(), (), (dom_wires, (), ()))

    @classmethod
    def caps(cls, left, right):
        adjoint = left.r if hasattr(left, "r") else left[::-1]
        if adjoint != right:
            raise AxiomError
        cod_wires = tuple(range(len(left))) + tuple(reversed(range(len(left))))
        return cls(
            cls.category.ob(), left @ right, (), ((), (), cod_wires))

    def transpose(self, left=False):
        """ The transpose of a hypergraph diagram. """
        return self.category.transpose(self, left)

    def rotate(self, left=False):
        """
        The half-turn rotation of a hypergraph, called with ``.l`` and ``.r``.
        """
        dom, cod = (x.l if left else x.r for x in (self.cod, self.dom))
        boxes = tuple(box.l if left else box.r for box in self.boxes[::-1])
        dom_wires = self.cod_wires[::-1]
        box_wires = tuple((x[::-1], y[::-1]) for x, y in self.box_wires[::-1])
        cod_wires = self.dom_wires[::-1]
        wires = dom_wires, box_wires, cod_wires
        return type(self)(
            dom, cod, boxes, wires, self.spider_types, self.offsets[::-1])

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))

    def explicit_trace(self, left=False):
        """
        The trace of a hypergraph with explicit boxes (trace, cup or cap).

        Parameters:
            left : Whether to trace on the left or right.

        Note
        ----
        When ``category.trace_factory`` is a subclass of ``category``,
        e.g. for symmetric diagrams, then the result is just one big trace box
        wrapped up as a hypergraph.

        Otherwise, we assume that the trace factory is a class method, e.g.
        for compact diagrams, in which case we use this method to introduce
        cup and cap boxes.
        """
        factory = self.category.trace_factory
        if isclass(factory) and issubclass(factory, self.category):
            return self.from_box(factory(self.to_diagram(), left))
        return factory.__func__(type(self), self, left)

    def trace(self, n=1, left=False):
        """
        The trace of a hypergraph is its pre- and post-composition with
        cups and caps to form a feedback loop.

        Parameters:
            n : The number of wires to trace.
            left : Whether to trace on the left or right.
        """
        assert_istraceable(self, n, left)
        dom, cod = (self.dom[n:], self.cod[n:]) if left\
            else (self.dom[:-n], self.cod[:-n])
        traced_wires = self.dom[:n] if left else self.dom[len(self.dom) - n:]
        traced_wires_r = traced_wires.r if hasattr(traced_wires, "r")\
            else traced_wires[::-1]
        return self.caps(traced_wires_r, traced_wires) @ dom\
            >> traced_wires_r @ self\
            >> self.cups(traced_wires_r, traced_wires) @ cod if left\
            else dom @ self.caps(traced_wires, traced_wires_r)\
            >> self @ traced_wires_r\
            >> cod @ self.cups(traced_wires, traced_wires_r)

    def interchange(self, i: int, j: int) -> Hypergraph:
        """
        Interchange boxes at indices ``i`` and ``j``.

        Parameters:
            i : The index of the first box.
            j : The index of the second box.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x = Ty('x')
        >>> f = Box('f', Ty(), x).to_hypergraph()
        >>> g = Box('g', x, Ty()).to_hypergraph()
        >>> print((f >> g).interchange(0, 1))
        Cap(x, x) >> g @ x >> f @ x >> Cup(x, x)
        """
        boxes, offsets = list(self.boxes), list(self.offsets)
        boxes[i], boxes[j] = boxes[j], boxes[i]
        offsets[i], offsets[j] = offsets[j], offsets[i]
        boxes, offsets = tuple(boxes), tuple(offsets)
        box_wires = list(self.box_wires)
        box_wires[i], box_wires[j] = box_wires[j], box_wires[i]
        wires = self.dom_wires, tuple(box_wires), self.cod_wires
        return type(self)(
            self.dom, self.cod, boxes, wires, self.spider_types, offsets)

    def simplify(self) -> Hypergraph:
        """
        Simplify by applying interchangers eagerly until the length of the
        diagram is minimal, takes quadratic time.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x = Ty('x')
        >>> f = Box('f', Ty(), x).to_hypergraph()
        >>> g = Box('g', x, Ty()).to_hypergraph()
        >>> assert (f >> g).interchange(0, 1).simplify() == f >> g
        """

        size = self.to_diagram().size
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes)):
                result = self.interchange(i, j)
                if result.to_diagram().size < size:
                    return result.simplify()
        return self

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise NotImplementedError

    def __eq__(self, other: Any):
        if not isinstance(other, Hypergraph):
            return False
        if not self.is_parallel(other):
            return False
        self_ok, other_ok = self._is_fast_eligible, other._is_fast_eligible
        if self_ok != other_ok:
            return False
        if self_ok:
            self_form = self._boundary_rooted_canonical()
            other_form = other._boundary_rooted_canonical()
            if self_form is not None and other_form is not None:
                return self_form == other_form
        return is_isomorphic(
            self.to_graph(), other.to_graph(), lambda x, y: x == y)

    def __hash__(self):
        if self._is_fast_eligible:
            return hash((
                self.dom, self.cod, len(self.boxes), self.n_spiders,
                tuple(sorted(repr(box) for box in self.boxes)),
                tuple(sorted(map(repr, self.spider_types)))))
        return hash((self.dom, self.cod, weisfeiler_lehman_graph_hash(
            self.to_graph(), node_attr="box")))

    def _box_dependencies(self):
        """
        Box-level dependency graph induced by the wiring, used to detect
        cycles and to compute a canonical box order in linear time, as an
        alternative to building the full port-level :meth:`causal_graph`.

        Returns a producer and a consumer box index for each spider
        (``-1`` for the global boundary), a mapping from each spider to its
        position in :attr:`dom_wires` and the resulting dependents/indegree
        lists for the boxes.
        """
        producer, consumer = [-1] * self.n_spiders, [-1] * self.n_spiders
        for i, (box_dom, box_cod) in enumerate(self.box_wires):
            for spider in box_dom:
                consumer[spider] = i
            for spider in box_cod:
                producer[spider] = i
        dom_port = {spider: i for i, spider in enumerate(self.dom_wires)}
        dependents = [[] for _ in self.boxes]
        indegree = [0] * len(self.boxes)
        for i, (box_dom, _) in enumerate(self.box_wires):
            for spider in box_dom:
                j = producer[spider]
                if j != -1:
                    dependents[j].append(i)
                    indegree[i] += 1
        return producer, consumer, dom_port, dependents, indegree

    def _is_acyclic_fast(self) -> bool:
        """
        Checks acyclicity of the box-level dependency graph with a plain
        Kahn's algorithm, avoiding the construction of a full networkx
        graph on ports.
        """
        _, _, _, dependents, indegree = self._box_dependencies()
        indegree, seen = list(indegree), 0
        ready = [i for i, d in enumerate(indegree) if d == 0]
        while ready:
            i = ready.pop()
            seen += 1
            for j in dependents[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append(j)
        return seen == len(self.boxes)

    @cached_property
    def is_boundary_connected(self) -> bool:
        """
        Checks boundary-connectedness, i.e. whether the graph with boxes as
        edges and spiders as vertices is connected, after adding one extra
        boundary node joined to every spider in :attr:`dom_wires` and
        :attr:`cod_wires`. The boundary node is always added, even when the
        boundary is empty, so that e.g. a closed diagram with a non-trivial
        interior (such as a trace of a tensor, or a scalar spider) is *not*
        boundary-connected: the boundary node then has no edges and is its
        own isolated component.

        Note this has nothing to do with acyclicity: there are acyclic
        diagrams that are not boundary-connected, e.g. a tensor of two
        scalar boxes, and there are boundary-connected diagrams that are
        cyclic, e.g. the trace of an endomorphism.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, x).to_hypergraph()
        >>> assert f.is_boundary_connected

        >>> # The trace of an endomorphism with a non-empty boundary is
        >>> # boundary-connected even though it is cyclic.
        >>> g = Box('g', x @ z, y @ z).to_hypergraph()
        >>> assert g.trace().is_boundary_connected

        >>> # Tracing away the whole domain leaves an empty, closed loop.
        >>> assert not f.trace().is_boundary_connected

        >>> scalar = Box('s', Ty(), Ty()).to_hypergraph()
        >>> assert not (scalar @ scalar).is_boundary_connected
        >>> assert not H.spiders(0, 0, x).is_boundary_connected

        >>> assert H.id(Ty()).is_boundary_connected
        """
        n_boxes = len(self.boxes)
        boundary = self.n_spiders + n_boxes
        parent = list(range(boundary + 1))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            i, j = find(i), find(j)
            if i != j:
                parent[i] = j

        for i, (box_dom, box_cod) in enumerate(self.box_wires):
            for spider in box_dom + box_cod:
                union(self.n_spiders + i, spider)
        for spider in self.dom_wires + self.cod_wires:
            union(boundary, spider)
        return len({find(i) for i in range(boundary + 1)}) == 1

    @cached_property
    def _is_fast_eligible(self) -> bool:
        """
        Whether :meth:`__eq__` and :meth:`__hash__` can use the linear-time
        boundary-rooted canonical form instead of graph isomorphism.
        """
        return self.is_monogamous and self.is_boundary_connected

    def _boundary_rooted_canonical(self) -> tuple | None:
        """
        Canonical form of a monogamous, boundary-connected hypergraph, used
        to compare such diagrams in linear time instead of checking for
        graph isomorphism. Computed by a deterministic breadth-first
        traversal rooted at the boundary, assigning each spider and box a
        canonical rank the first time it is reached; this tolerates cycles
        natively, unlike a topological sort.

        Returns ``None`` if the traversal could not certify a complete
        canonical order, e.g. because of a genuine symmetry between two
        independent parts of the diagram, in which case the caller should
        fall back to a slower method.
        """
        producer, consumer, _, _, _ = self._box_dependencies()
        canon: dict[int, int] = {}
        order: list[int] = []

        def see(spider):
            if spider not in canon:
                canon[spider] = len(order)
                order.append(spider)

        for spider in self.dom_wires:
            see(spider)
        for spider in self.cod_wires:
            see(spider)

        box_rank = [None] * len(self.boxes)
        next_rank = 0
        position = 0
        while position < len(order):
            spider = order[position]
            position += 1
            for box in (producer[spider], consumer[spider]):
                if box != -1 and box_rank[box] is None:
                    box_rank[box] = next_rank
                    next_rank += 1
                    box_dom, box_cod = self.box_wires[box]
                    for s in box_dom:
                        see(s)
                    for s in box_cod:
                        see(s)

        if next_rank != len(self.boxes):
            return None
        if len(order) != self.n_spiders - len(self.scalar_spiders):
            return None

        box_order = sorted(range(len(self.boxes)), key=box_rank.__getitem__)
        return (
            self.dom, self.cod,
            tuple(canon[s] for s in self.dom_wires),
            tuple(canon[s] for s in self.cod_wires),
            tuple(
                (self.boxes[i],
                 tuple(canon[s] for s in self.box_wires[i][0]),
                 tuple(canon[s] for s in self.box_wires[i][1]))
                for i in box_order),
            tuple(self.spider_types[order[c]] for c in range(len(order))))

    def __repr__(self):
        spider_types = f", spider_types={self.spider_types}"\
            if self.scalar_spiders else ""
        return factory_name(type(self))\
            + f"(dom={repr(self.dom)}, cod={repr(self.cod)}, " \
              f"boxes={repr(self.boxes)}, " \
              f"wires={repr(self.wires)}{spider_types})"

    def __str__(self):
        return str(self.to_diagram())

    @property
    def bijection(self):
        """
        Bijection between ports.

        Raises
        ------
            ValueError : If the hypergraph is not bijective.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> for i, port in enumerate(f.ports): print(i, port)
        0 Node('input', i=0, obj=x)
        1 Node('dom', depth=0, i=0, obj=x)
        2 Node('cod', depth=0, i=0, obj=y)
        3 Node('output', i=0, obj=y)
        >>> for i, j in enumerate(f.bijection): print(f"{i} -> {j}")
        0 -> 1
        1 -> 0
        2 -> 3
        3 -> 2
        """
        if not self.is_bijective:
            raise ValueError
        result, flat_wires = {}, list(self.flat_wires)
        for i, spider in enumerate(flat_wires):
            if spider not in flat_wires[i + 1:]:
                continue
            j = flat_wires[i + 1:].index(spider) + i + 1
            result[i], result[j] = j, i
        return [result[i] for i in sorted(result)]

    def _hypergraph_to_canonical(self) -> tuple[int, ...]:
        """ Relabel hypergraph port order to canonical map port order. """
        result = list(range(len(self.dom)))
        start = len(self.dom)
        for box in self.boxes:
            result.extend(start + i for i in range(len(box.dom)))
            result.extend(
                start + len(box.dom) + len(box.cod) - i - 1
                for i in range(len(box.cod)))
            start += len(box.dom @ box.cod)
        result.extend(range(start, start + len(self.cod)))
        return tuple(result)

    @classmethod
    def from_map(cls, old) -> Hypergraph:
        """
        Forget orientation and return the underlying bijective hypergraph
        given by the edge permutation.

        >>> from discopy.compact import Ty, Box, CMap, Hypergraph
        >>> x, y = map(Ty, "xy")
        >>> cmap = Box("f", x, y).to_map()
        >>> Hypergraph.from_map(cmap).to_map() == cmap
        True
        """
        hypergraph_to_canonical = list(range(len(old.dom)))
        start = len(old.dom)
        for box in old.boxes:
            hypergraph_to_canonical.extend(
                start + i for i in range(len(box.dom)))
            hypergraph_to_canonical.extend(
                start + len(box.dom) + len(box.cod) - i - 1
                for i in range(len(box.cod)))
            start += len(box.dom @ box.cod)
        hypergraph_to_canonical.extend(range(start, start + len(old.cod)))
        hypergraph_to_canonical = tuple(hypergraph_to_canonical)
        canonical_to_hypergraph = Permutation(hypergraph_to_canonical).dagger()
        edges = old.edges.conjugate(canonical_to_hypergraph)
        spider_types, flat_wires = [], [None] * old.n_ports
        for i in range(old.n_ports):
            j = edges[i]
            if i > j:
                continue
            spider = len(spider_types)
            spider_types.append(old.ports[hypergraph_to_canonical[i]].obj)
            flat_wires[i] = flat_wires[j] = spider
        spider_types.extend(old.loops)
        wires = cls.rebracket(
            None, flat_wires, dom=old.dom, boxes=old.boxes)
        factory = cls[old.category]
        return factory(
            old.dom, old.cod, old.boxes, wires,
            tuple(spider_types), old.offsets)

    def to_map(self):
        """
        Build a combinatorial map, introducing spider boxes if needed.
        """
        if not self.is_bijective:
            return self.make_bijective().to_map()
        factory = getattr(self.category, "map_factory", None)
        if factory is None:
            factory = cmap.CMap[type(self).category]
        relabeling = Permutation(self._hypergraph_to_canonical())
        edges = Permutation(self.bijection).conjugate(relabeling)
        loops = tuple(self.spider_types[i] for i in self.scalar_spiders)
        return factory(
            self.dom, self.cod, self.boxes, edges, offsets=self.offsets,
            loops=loops)

    @property
    def is_bijective(self) -> bool:
        """
        Checks bijectivity, i.e. each spider is connected to two or zero ports.
        In that case, the diagram actually lives in a compact-closed category,
        i.e. it can be drawn using only swaps, cups and caps.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> assert f.is_bijective and f.transpose().is_bijective
        >>> assert H.cups(x, x).is_bijective and H.caps(x, x).is_bijective
        >>> assert H.spiders(0, 0, x).is_bijective
        >>> assert not H.spiders(1, 2, x).is_bijective
        """
        return all(len(x | y) in [0, 2] for x, y in self.spider_wires)

    @property
    def is_monogamous(self) -> bool:
        """
        Checks monogamy, i.e. each input connects to exactly one output,
        formally whether :code:`self.wires` induces a bijection::

            len(self.dom) + sum(len(box.dom) for box in boxes)
            == self.n_spiders - len(self.scalar_spiders)
            == sum(len(box.cod) for box in boxes) + len(self.dom)

        In that case, the diagram actually lives in a traced category.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()

        >>> assert f.is_monogamous
        >>> assert (f >> f[::-1]).is_monogamous

        >>> assert H.spiders(0, 0, x).is_monogamous

        >>> cycle = H.caps(x, x) >> x @ (f >> f[::-1]) >> H.cups(x, x)
        >>> assert cycle.is_monogamous

        >>> assert not f.transpose().is_monogamous
        >>> assert not H.cups(x, x).is_monogamous
        >>> assert not H.spiders(1, 2, x).is_monogamous
        >>> assert not H.spiders(2, 3, x).is_monogamous
        """
        for input_wires, output_wires in self.spider_wires:
            if len(input_wires) != len(output_wires):
                return False
            if len(input_wires) + len(output_wires) not in [0, 2]:
                return False
        return True

    @property
    def is_left_monogamous(self) -> bool:
        """
        Checks left monogamy, i.e. if each non-scalar spider is connected
        to exactly one input port.
        """
        return all(len(x) == 1 for x, y in self.spider_wires if x.union(y))

    @property
    def is_causal(self) -> bool:
        """
        Checks causality, i.e. if each non-scalar spider is connected to
        exactly one input port, there is no directed cycle, and wires point
        forward in the current port order. It is equivalent to:
        - is_left_monogamous
        - is_acyclic
        - is_topological_ordered

        If the diagram is causal then it lives in a symmetric monoidal
        category with a supply of commutative comonoids.

        If the diagram is causal and monogamous then it actually lives in
        a symmetric monoidal category, i.e. it can be drawn using only swaps.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> assert f.is_causal
        >>> assert (f >> H.spiders(1, 0, y)).is_causal
        >>> assert (H.spiders(1, 2, x) >> f @ f).is_causal

        >>> g = Box('g', y, z).to_hypergraph()
        >>> assert (f >> g).interchange(0, 1).is_acyclic
        >>> assert not (f >> g).interchange(0, 1).is_causal

        >>> cycle = H.caps(x, x) >> H.cups(x, x)
        >>> assert not cycle.is_acyclic
        >>> assert not cycle.is_causal
        >>> assert cycle.make_causal().is_causal

        >>> assert not H.cups(x, x).is_causal
        """
        return all(len(input_wires) == 1 and all(
            u < v for u in input_wires for v in output_wires)
            for input_wires, output_wires in self.spider_wires)

    @property
    def is_acyclic(self) -> bool:
        """
        Checks that the causal graph has no directed cycle.
        As an edge case, we also need to check that there are no scalar
        spiders, otherwise the causal graph has 0 nodes and is thus trivially
        acyclic.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Cap, Cup, Hypergraph as H
        >>> x, y, z = map(Ty, 'xyz')
        >>> f = Box('f', x, x)
        >>> g = Box('f', x @ z, y @ z)

        >>> # Simple case: cup and caps and trace form cycles
        >>> assert not (
        ...     Cap(x, x) >> x @ f >> Cup(x, x)
        ... ).to_hypergraph().is_acyclic
        >>> assert not g.trace().to_hypergraph().is_acyclic

        >>> # Breaking causality but not acyclicity:
        >>> f_snake = (
        ...     Cap(x, x) @ x >> x @ f @ x >> x @ Cup(x, x)
        ... ).to_hypergraph()
        >>> assert not f_snake.is_causal and f_snake.is_acyclic

        >>> # Edge case: cyclic hypergraph without boxes
        >>> assert not (Cap(x, x) >> Cup(x, x)).to_hypergraph().is_acyclic
        """
        return not self.scalar_spiders and self._is_acyclic_fast()

    @property
    def is_topologically_ordered(self) -> bool:
        """
        Checks that causal wires point forward in the current port order.
        """
        return all(all(
            u < v for u in input_wires for v in output_wires)
            for input_wires, output_wires in self.spider_wires)

    def causal_graph(self) -> Graph:
        """
        Directed graph on ports induced by spiders and boxes.
        """
        graph = Graph()
        graph.add_nodes_from(range(len(self.ports)))
        for input_wires, output_wires in self.spider_wires:
            graph.add_edges_from(
                (source, target)
                for source in input_wires for target in output_wires)
        port = len(self.dom)
        for box in self.boxes:
            dom_ports = range(port, port + len(box.dom))
            cod_ports = range(
                port + len(box.dom),
                port + len(box.dom @ box.cod)
            )
            graph.add_edges_from(
                (source, target)
                for source in dom_ports for target in cod_ports)
            port += len(box.dom @ box.cod)
        return graph

    def topological_order(self) -> Hypergraph:
        """
        Reorder boxes so that causal wires point forward in ``flat_wires``.
        """
        graph = Graph()
        graph.add_nodes_from(range(len(self.boxes)))
        for input_wires, output_wires in self.spider_wires:
            if len(input_wires) != 1:
                continue
            input_port, = input_wires
            input_node = self.ports[input_port]
            input_box = getattr(input_node, "depth", None)\
                if input_node.kind == "cod" else None
            for output_port in output_wires:
                output_node = self.ports[output_port]
                output_box = getattr(output_node, "depth", None)\
                    if output_node.kind == "dom" else None
                if input_box is not None and output_box is not None\
                        and input_box != output_box:
                    graph.add_edge(input_box, output_box)
        order = tuple(topological_sort(graph))
        if order == tuple(range(len(self.boxes))):
            return self
        boxes = tuple(self.boxes[i] for i in order)
        box_wires = tuple(self.box_wires[i] for i in order)
        offsets = tuple(self.offsets[i] for i in order)
        return type(self)(
            self.dom, self.cod, boxes, (self.dom_wires, box_wires,
                                        self.cod_wires),
            self.spider_types, offsets)

    def make_bijective(self) -> Hypergraph:
        """
        Introduces spider boxes to make self bijective.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider, Hypergraph as H

        >>> spider = H.spiders(3, 2, Ty('x')).make_bijective()
        >>> assert spider.boxes == (Spider(3, 2, Ty('x')), )
        >>> assert spider.wires == ((0, 1, 2), (((0, 1, 2), (3, 4)),), (3, 4))

        >>> copy = H.spiders(1, 2, Ty('x', 'y')).make_bijective()
        >>> assert copy.boxes == (Spider(1, 2, Ty('x')), Spider(1, 2, Ty('y')))

        >>> unit = H.spiders(0, 1, Ty('x', 'y')).make_bijective()
        >>> assert unit.boxes == (Spider(0, 1, Ty('y')), Spider(0, 1, Ty('x')))
        """
        boxes = list(self.boxes)
        f_wires = list(self.flat_wires)
        spider_types = list(self.spider_types)
        for spider, (typ, (input_wires, output_wires)) in reversed(list(
                enumerate(zip(spider_types, self.spider_wires)))):
            n_legs = len(input_wires) + len(output_wires)
            if n_legs in [0, 2]:
                continue
            if input_wires:
                node = self.ports[max(input_wires)]
                depth = 0 if node.kind == "input" else node.depth + 1
            else:
                node = self.ports[min(output_wires)]
                depth = len(boxes) if node.kind == "output" else node.depth
            boxes = boxes[:depth] + [self.category.spider_factory(
                len(input_wires), len(output_wires), typ)] + boxes[depth:]
            offsets = self.offsets[:depth] + (None, ) + self.offsets[depth:]
            port_key = lambda port: (
                getattr(self.ports[port], "i", port), port)
            ports = tuple(sorted(input_wires, key=port_key))\
                + tuple(sorted(output_wires, key=port_key))
            for j, port in enumerate(ports):
                f_wires[port] = len(spider_types) + j
            i = len(self.dom) + len(
                sum([sum(ports, ()) for ports in self.box_wires[:depth]], ()))
            f_wires = f_wires[:i] + list(range(
                len(spider_types), len(spider_types) + n_legs)) + f_wires[i:]
            spider_types += n_legs * [typ]
            del spider_types[spider]
            f_wires = [w - 1 if w > spider else w for w in f_wires]
            wires = self.rebracket(f_wires, boxes=boxes)
            return type(self)(
                self.dom, self.cod, tuple(boxes),
                wires, spider_types, offsets).make_bijective()
        assert self.is_bijective
        return self

    def make_monogamous(self) -> Hypergraph:
        """
        Introduce :class:`Cup` and :class:`Cap` boxes to make self monogamous.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Cup, Cap, Spider
        >>> x = Ty('x')
        >>> h = Box('f', x, x).transpose().to_hypergraph().make_monogamous()
        >>> assert list(zip(h.boxes, h.box_wires)) == [
        ...     (Cap(x, x),      ((),     (1, 2))),
        ...     (Box('f', x, x), ((1,),   (3,)  )),
        ...     (Cup(x, x),      ((0, 3), ()    ))]
        """
        if not self.is_bijective:
            return self.make_bijective().make_monogamous()
        for kinds, cups_or_caps in [
                (["input", "cod"], "cups"),
                (["dom", "output"], "caps")]:
            for source, spider in [
                    (source, spider) for source, (spider, port)
                    in enumerate(zip(self.flat_wires, self.ports))
                    if port.kind in kinds]:
                if spider not in self.flat_wires[source + 1:]:
                    continue
                target = (
                    self.flat_wires[source + 1:].index(spider) + source + 1)
                if self.ports[target].kind in kinds:
                    spider_types = dict(enumerate(self.spider_types))
                    typ = spider_types[spider]
                    source_obj, target_obj = (
                        self.ports[source].obj, self.ports[target].obj)
                    left, right = len(spider_types), len(spider_types) + 1
                    fwires = list(self.flat_wires)
                    fwires[source], fwires[target] = left, right
                    if cups_or_caps == "cups":
                        boxes = self.boxes + (self.category.cup_factory(
                            source_obj, target_obj), )
                        offsets = self.offsets + (None, )
                        fwires = fwires[:len(fwires) - len(self.cod)] + [
                            left, right] + fwires[len(fwires) - len(self.cod):]
                    else:
                        boxes = (self.category.cap_factory(
                            source_obj, target_obj), ) + self.boxes
                        offsets = (None, ) + self.offsets
                        fwires = fwires[:len(self.dom)] + [
                            left, right] + fwires[len(self.dom):]
                    wires = self.rebracket(fwires, boxes=boxes)
                    spider_types[left] = spider_types[right] = typ
                    del spider_types[spider]
                    return type(self)(
                        self.dom, self.cod, boxes, wires, spider_types, offsets
                    ).make_monogamous()
        assert self.is_monogamous
        return self

    def make_left_monogamous(self) -> Hypergraph:
        """
        Introduce spider boxes to make self left monogamous.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H, Spider
        >>> h = H.spiders(2, 3, Ty('x')).make_left_monogamous()
        >>> assert h.boxes == (Spider(2, 1, Ty('x')), )
        >>> assert h.wires == ((0, 1), (((0, 1), (2,)),), (2, 2, 2))
        """
        for spider, (typ, (input_wires, output_wires)) in enumerate(
                zip(self.spider_types, self.spider_wires)):
            if len(input_wires) == 1:
                continue
            depth = getattr(self.ports[max(input_wires)], "depth", -1) + 1\
                if input_wires else 0
            boxes = self.boxes[:depth] + (self.category.spider_factory(
                len(input_wires), 1, typ), ) + self.boxes[depth:]
            offsets = self.offsets[:depth] + (None, ) + self.offsets[depth:]
            fwires = list(self.flat_wires)
            for j, port in enumerate(input_wires):
                fwires[port] = self.n_spiders + j
            i = len(self.dom) + len(
                sum([sum(wires, ()) for wires in self.box_wires[:depth]], ()))
            fwires = fwires[:i] + list(range(
                self.n_spiders, self.n_spiders + len(input_wires))
            ) + [spider] + fwires[i:]
            wires = self.rebracket(fwires, boxes=boxes)
            spider_types = self.spider_types + len(input_wires) * (typ, )
            return type(self)(
                self.dom, self.cod, boxes, wires, spider_types, offsets
            ).make_left_monogamous()
        assert self.is_left_monogamous
        return self

    def make_causal(self) -> Hypergraph:
        """
        Introduce trace boxes to make self causal.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Cup, Cap
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x).to_hypergraph()
        >>> assert f.trace().make_causal().boxes\\
        ...     == (Cap(x, x), f.boxes[0], Cup(x, x))

        >>> from discopy.frobenius import Hypergraph as H, Spider
        >>> assert H.spiders(2, 1, x).make_causal().boxes\\
        ...     == (Spider(2, 1, x),)
        """
        if not self.is_left_monogamous:
            return self.make_left_monogamous().make_causal()
        for input_spider, (typ, (input_wires, output_wires)) in enumerate(
                zip(self.spider_types, self.spider_wires)):
            if input_wires:
                continue
            assert not output_wires
            dom, cod, boxes = self.dom @ typ, self.cod @ typ, self.boxes
            dom_wires = self.dom_wires + (input_spider, )
            cod_wires = self.cod_wires + (input_spider, )
            wires = (dom_wires, self.box_wires, cod_wires)
            arg = type(self)(
                dom, cod, boxes, wires, self.spider_types, self.offsets)
            return arg.make_causal().explicit_trace()
        if self.is_causal:
            return self
        causal_graph = self.causal_graph()
        for input_spider, (typ, (input_wires, output_wires)) in enumerate(
                zip(self.spider_types, self.spider_wires)):
            input_wire, = input_wires
            for output_wire in output_wires:
                if input_wire < output_wire and\
                        not has_path(causal_graph, output_wire, input_wire):
                    continue
                dom, cod = self.dom @ typ, self.cod @ typ
                spider_types = self.spider_types + (typ, )
                output_spider = len(spider_types) - 1
                fwires = list(self.flat_wires)
                fwires[output_wire] = output_spider
                fwires = fwires[:len(self.dom)] + [output_spider]\
                    + fwires[len(self.dom):] + [input_spider]
                wires = self.rebracket(fwires, dom=dom)
                arg = type(self)(
                    dom, cod, self.boxes, wires, spider_types, self.offsets)
                return arg.make_causal().explicit_trace()
        assert self.is_causal
        return self

    @property
    def is_generator(self):
        """ Whether the hypergraph is a single generator. """
        return len(self.boxes) == 1 and self == self.from_box(self.boxes[0])

    @property
    def generator(self):
        """ Return the `f` from `Hypergraph.from_box(f)` if `is_generator`. """
        return self.boxes[0] if self.is_generator else None

    @classmethod
    def from_box(cls, box: Box) -> Hypergraph:
        """
        Turn a box into a hypergraph with binary spiders for each wire.

        Parameters:
            box : The box to turn into a hypergraph.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y, z = map(Ty, "xyz")
        >>> for p in Box('f', x, y @ z).to_hypergraph().ports: print(p)
        Node('input', i=0, obj=x)
        Node('dom', depth=0, i=0, obj=x)
        Node('cod', depth=0, i=0, obj=y)
        Node('cod', depth=0, i=1, obj=z)
        Node('output', i=0, obj=y)
        Node('output', i=1, obj=z)
        """
        spider_types = tuple(box.dom @ box.cod)
        left = tuple(range(len(box.dom)))
        right = tuple(range(len(box.dom), len(box.dom @ box.cod)))
        wires = (left, ((left, right), ), right)
        return cls(box.dom, box.cod, (box, ), wires, spider_types)

    @classmethod
    def from_diagram(cls, old: Diagram) -> Hypergraph:
        """
        Turn a :class:`Diagram` into a :class:`Hypergraph`.

        Parameters:
            old : The planar diagram to encode as hypergraph.

        Example
        -------
        >>> from discopy.frobenius import Ty, Hypergraph as H
        >>> x, y = map(Ty, "xy")
        >>> back_n_forth = lambda d: H.from_diagram(d.to_diagram())
        >>> for d in [H.spiders(0, 0, x),
        ...           H.spiders(2, 3, x),
        ...           H.spiders(1, 2, x @ y)]:
        ...     assert back_n_forth(d) == d
        """
        factory = cls[type(old).ar]
        return factory.functor(
            ob_map=lambda typ: typ, ar_map=cls.from_box,
            dom=type(old), cod=cls)(old)

    def to_diagram(self) -> Diagram:
        """
        Downgrade to :class:`Diagram`, called by :code:`print`.

        Note
        ----
        Hypergraphs can be translated to planar diagrams in two different ways:

        * either we first :meth:`make_bijective` (introducing spiders) then
          :meth:`make_monogamous` (introducing cups and caps) and finally
          :meth:`make_causal` (introducing traces)
        * or we first :meth:`make_left_monogamous` (introducing merges) then
          :meth:`make_causal` (introducing traces) and finally
          :meth:`make_bijective` (introducing copies).

        We use the first order whenever :attr:`category` is a subclass of
        :class:`HypergraphCategory`, i.e. it has a self-dual supply of spiders,
        or when it has cups and caps. Otherwise, if it is a subclass of
        :class:`MarkovCategory`, i.e. it has a supply of comonoids, we use the
        second order and introduce copies. When the category has neither, a
        non-monogamous hypergraph cannot be downgraded and we raise an error.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x = Ty('x')
        >>> v = Box('v', Ty(), x @ x).to_hypergraph()
        >>> print(v >> H.swap(x, x) >> v[::-1])
        v >> Swap(x, x) >> v[::-1]
        >>> print(x @ H.swap(x, x) >> v[::-1] @ x)
        x @ Swap(x, x) >> v[::-1] @ x
        """
        if self.scalar_spiders or not self.is_causal or not self.is_monogamous:
            if issubclass(self.category, HypergraphCategory):
                return self.make_monogamous().make_causal().to_diagram()
            if issubclass(self.category, MarkovCategory):
                return self.make_causal().make_bijective().to_diagram()
            if not self.is_monogamous and getattr(
                    self.category, "cup_factory", None) is None:
                raise AxiomError(messages.NO_STRUCTURE_TO_DOWNGRADE.format(
                    factory_name(self.category)))
            return self.make_monogamous().make_causal().to_diagram()
        diagram, scan = self.category.id(self.dom), self.dom_wires
        for depth, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            dom_wires, cod_wires = self.box_wires[depth]
            for i, obj in enumerate(box.dom):
                j = scan.index(dom_wires[i])
                if i == 0 and offset is None:
                    offset = j
                elif j > offset + i:
                    diagram >>= diagram.cod[:offset + i] @ diagram.swap(
                        diagram.cod[offset + i:j], diagram.cod[j]
                    ) @ diagram.cod[j + 1:]
                    scan = (scan[:offset + i] + scan[j:j + 1]) + (
                        scan[offset + i:j] + scan[j + 1:])
                elif j < offset + i:
                    diagram >>= diagram.cod[:j] @ diagram.swap(
                        diagram.cod[j], diagram.cod[j + 1:offset + i]
                    ) @ diagram.cod[offset + i:]
                    scan = (scan[:j] + scan[j + 1:offset + i]) + (
                        scan[j:j + 1] + scan[offset + i:])
                    offset -= 1
                assert len(scan) == len(diagram.cod)
            offset = 0 if offset is None else offset
            scan = scan[:offset] + cod_wires + scan[offset + len(box.dom):]
            diagram >>= diagram.cod[:offset] @ box @ diagram.cod[
                offset + len(box.dom):]
        for i, spider in enumerate(self.cod_wires):
            j = scan.index(spider)
            if i < j:
                diagram >>= diagram.cod[:i] @ diagram.swap(
                    diagram.cod[i:j], diagram.cod[j:j + 1]
                ) @ diagram.cod[j + 1:]
                scan = scan[:i] + scan[j:j + 1] + scan[i:j] + scan[j + 1:]
        return diagram

    @classmethod
    def from_callable(cls, dom: Ty, cod: Ty) -> Callable[Callable, Hypergraph]:
        """
        Turns an arbitrary Python function into a causal hypergraph.

        Parameters:
            dom : The domain of the hypergraph.
            cod : The codomain of the hypergraph.
        """
        def decorator(func):
            graph, box_nodes, spider_nodes = Graph(), [], []

            def apply(box, *inputs, offset=None):
                for node in inputs:
                    assert_isinstance(node, Node)
                if len(inputs) != len(box.dom):
                    raise AxiomError(f"Expected {len(box.dom)} inputs, "
                                     f"got {len(inputs)} instead.")
                depth = len(box_nodes)
                box_node = Node("box", box=box, depth=depth, offset=offset)
                box_nodes.append(box_node)
                graph.add_node(box_node)
                for i, obj in enumerate(box.dom):
                    if inputs[i].obj != obj:
                        raise AxiomError(f"Expected {obj} as input, "
                                         f"got {inputs[i].obj} instead.")
                    dom_node = Node("dom", obj=obj, i=i, depth=depth)
                    graph.add_edge(inputs[i], dom_node)
                    graph.add_edge(dom_node, box_node)
                outputs = []
                for i, obj in enumerate(box.cod):
                    cod_node = Node("cod", obj=obj, i=i, depth=depth)
                    spider = Node("spider", obj=obj, i=len(spider_nodes))
                    graph.add_edge(box_node, cod_node)
                    graph.add_edge(cod_node, spider)
                    spider_nodes.append(spider)
                    outputs.append(spider)
                return untuplify(outputs)

            cls.category.__call__ = apply
            for i, obj in enumerate(dom):
                input_node = Node("input", obj=obj, i=i)
                input_spider = Node("spider", obj=obj, i=i)
                spider_nodes.append(input_spider)
                graph.add_edge(input_node, input_spider)
            for i, spider in enumerate(tuplify(func(*spider_nodes))):
                assert_isinstance(spider, Node)
                node = Node("output", obj=spider.obj, i=i)
                graph.add_edge(spider, node)
            del cls.category.__call__
            result = cls.from_graph(graph)
            if result.cod != cod:
                raise AxiomError(f"Expected diagram.cod == {cod}, "
                                 f"got {result.cod} instead.")
            return result

        return decorator

    @classmethod
    def from_graph(cls, graph: Graph) -> Hypergraph:
        """ The inverse of :meth:`to_graph`. """
        def predecessor(node):
            result, = graph.predecessors(node)
            return result

        def successor(node):
            result, = graph.successors(node)
            return result

        inputs, outputs, box_nodes, spider_nodes = [], [], [], []
        for node in graph.nodes:
            for kind, nodelist in zip(
                    ["input", "output", "box", "spider"],
                    [inputs, outputs, box_nodes, spider_nodes]):
                if node.kind == kind:
                    nodelist.append(node)
        dom = sum([n.obj for n in inputs], cls.category.ob())
        cod = sum([n.obj for n in outputs], cls.category.ob())
        boxes = tuple(n.box for n in box_nodes)
        offsets = tuple(n.offset for n in box_nodes)
        spider_types = {n: n.obj for n in spider_nodes}
        wires = tuple(map(successor, sorted(inputs, key=lambda node: node.i)))
        for box_node in box_nodes:
            wires += tuple(map(predecessor, sorted(
                graph.predecessors(box_node), key=lambda node: node.i)))
            wires += tuple(map(successor, sorted(
                graph.successors(box_node), key=lambda node: node.i)))
        wires += tuple(map(
            predecessor, sorted(outputs, key=lambda node: node.i)))
        wires = Hypergraph.rebracket(None, wires, dom=dom, boxes=boxes)
        return cls(dom, cod, boxes, wires, spider_types, offsets)

    def to_graph(self) -> Graph:
        """
        Translate a hypergraph into a labeled graph with nodes for inputs,
        outputs, boxes, domain, codomain and spiders.
        """
        graph = Graph()
        graph.add_nodes_from(
            (Node("spider", i=i, obj=obj), dict(box=None))
            for i, obj in enumerate(self.spider_types))
        graph.add_nodes_from(
            (Node("input", i=i, obj=obj), dict(i=i, box=None))
            for i, obj in enumerate(self.dom))
        graph.add_edges_from(
            (Node("input", i=i, obj=obj), Node("spider", i=j, obj=obj))
            for i, (j, obj) in enumerate(
                zip(self.dom_wires, self.dom)))
        for i, (box, (dom_wires, cod_wires)) in enumerate(
                zip(self.boxes, self.box_wires)):
            box_node = Node("box", box=box, i=i)
            graph.add_node(box_node, box=box)
            for case, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    obj = self.spider_types[spider]
                    spider_node = Node("spider", i=spider, obj=obj)
                    port_node = Node(case, i=i, j=j)
                    graph.add_node(port_node, j=j, box=None)
                    if case == "dom":
                        graph.add_edge(spider_node, port_node)
                        graph.add_edge(port_node, box_node)
                    else:
                        graph.add_edge(box_node, port_node)
                        graph.add_edge(port_node, spider_node)
        graph.add_nodes_from(
            (Node("output", i=i, obj=obj), dict(i=i, box=None))
            for i, obj in enumerate(self.cod))
        graph.add_edges_from(
            (Node("spider", i=j, obj=obj), Node("output", i=i, obj=obj))
            for i, (j, obj) in enumerate(zip(self.cod_wires, self.cod)))
        return graph

    def depth(self) -> int:
        """ The depth of a causal hypergraph. """
        return dag_longest_path_length(self.make_causal().to_graph()) // 4

    def spring_layout(self, seed=None, k=None):
        """ Computes a layout using a force-directed algorithm. """
        if seed is not None:
            random.seed(seed)
        graph, pos = self.to_graph().to_undirected(), {}
        height = len(self.boxes) + self.n_spiders
        width = max(len(self.dom), len(self.cod))
        for i, obj in enumerate(self.dom):
            pos[Node("input", i=i, obj=obj)] = (i, height)
        for i, (dom_wires, cod_wires) in enumerate(self.box_wires):
            box_node = Node("box", i=i, box=self.boxes[i])
            pos[box_node] = (
                random.uniform(-width / 2, width / 2),
                random.uniform(0, height))
            for kind, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    pos[Node(kind, i=i, j=j)] = pos[box_node]
        for i, obj in enumerate(self.spider_types):
            pos[Node("spider", i=i, obj=obj)] = (
                random.uniform(-width / 2, width / 2),
                random.uniform(0, height))
        for i, obj in enumerate(self.cod):
            pos[Node("output", i=i, obj=obj)] = (i, 0)
        fixed = self.ports[:len(self.dom)] + self.ports[
            len(self.ports) - len(self.cod):] or None
        pos = spring_layout(graph, pos=pos, fixed=fixed, k=k, seed=seed)
        return graph, pos

    def draw(self, seed=None, k=.25, path=None):
        """
        Draw a hypegraph using a force-based layout algorithm.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ z).to_hypergraph()
        >>> f.draw(
        ...     path='docs/_static/hypergraph/box.png', seed=42)

        .. image:: /_static/hypergraph/box.png
            :align: center

        >>> (H.spiders(2, 2, x) >> f @ x).draw(
        ...     path='docs/_static/hypergraph/diagram.png', seed=42)

        .. image:: /_static/hypergraph/diagram.png
            :align: center
        """
        graph, pos = self.spring_layout(seed=seed, k=k)
        for i, (box, (dom_wires, cod_wires)) in enumerate(
                zip(self.boxes, self.box_wires)):
            box_node = Node("box", i=i, box=box)
            for kind, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    port_node = Node(kind, i=i, j=j)
                    x, y = pos[box_node]
                    if not getattr(box, "draw_as_spider", False):
                        y += .25 if kind == "dom" else -.25
                        x -= .25 * (len(wires[:-1]) / 2 - j)
                    pos[port_node] = x, y
        labels = {
            node: self.spider_types[node.i] if node.kind == "spider"
            else self.boxes[node.i].name if node.kind == "box" else ""
            for node in graph.nodes}
        nodelist = list(graph.nodes)
        node_size = [
            300 if node.kind in ["spider", "box"] else 0 for node in nodelist]
        draw_networkx(
            graph, pos=pos, labels=labels,
            nodelist=nodelist, node_size=node_size,
            node_color="white", edgecolors="black")
        if path is not None:
            plt.savefig(path)
            plt.close()
        plt.show()
