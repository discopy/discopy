# -*- coding: utf-8 -*-

"""
The free hypergraph category with cospans of labeled hypergraphs as arrows.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Hypergraph

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        pushout
        Id
        Box
        Swap
        Spider
        Cup
        Cap
"""

from __future__ import annotations

from collections.abc import Callable

import random

import matplotlib.pyplot as plt

from networkx import Graph, spring_layout, draw_networkx
from networkx.algorithms.isomorphism import is_isomorphic

from discopy import cat, monoidal, drawing
from discopy.cat import factory, AxiomError, Composable
from discopy.drawing import Node
from discopy.monoidal import (
    Ty, Box, Category, Functor, Whiskerable, assert_isatomic)
from discopy.utils import (
    factory_name,
    assert_isinstance,
    pushout,
    mmap,
    NamedGeneric,
)


class Hypergraph(Composable, Whiskerable, NamedGeneric['category']):
    """
    Hypergraph in a hypergraph category.

    Parameters:
        dom (frobenius.Ty) : The domain of the diagram, i.e. its input.
        cod (frobenius.Ty) : The codomain of the diagram, i.e. its output.
        boxes (tuple[frobenius.Box, ...]) : The boxes inside the diagram.
        wires (tuple[Any]) : List of wires from ports to spiders.
        spider_types : Mapping[Any, frobenius.Ty]
            Mapping from spiders to atomic types, if :code:`None` then this is
            computed from the types of ports.

    Note
    ----
    The wires go from ports to spiders, they are given as a list of length::

        len(dom) + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod)

    The values themselves don't matter, they are simply labels for the spiders.
    We must have :code:`len(types) >= len(set(wires))`, the spiders that don't
    appear as the target of a wire are scalar spiders, i.e. with zero legs.

    Abstractly, a hypergraph diagram can be seen as a cospan for the boundary::

        range(len(dom)) -> range(n_spiders) <- range(len(cod))

    together with a cospan for each box in boxes::

        range(len(box.dom)) -> range(n_spiders) <- range(len(box.cod))

    Composition of two hypergraph diagram is given by the pushout of the span::

        range(self.n_spiders) <- range(len(self.cod)) -> range(other.n_spiders)

    Examples
    --------
    >>> from discopy.frobenius import Ty, Box, Hypergraph as H

    >>> x, y, z = map(Ty, "xyz")

    >>> assert H.id(x @ y @ z).n_spiders == 3
    >>> assert H.id(x @ y @ z).wires == [0, 1, 2, 0, 1, 2]

    >>> assert H.swap(x, y).n_spiders == 2
    >>> assert H.swap(x, y).wires == [0, 1, 1, 0]

    >>> assert H.spiders(1, 2, x @ y).n_spiders == 2
    >>> assert H.spiders(1, 2, x @ y).wires == [0, 1, 0, 1, 0, 1]
    >>> assert H.spiders(0, 0, x @ y @ z).n_spiders == 3
    >>> assert H.spiders(0, 0, x @ y @ z).wires == []

    >>> f, g = Box('f', x, y).to_hypergraph(), Box('g', y, z).to_hypergraph()

    >>> assert f.n_spiders == g.n_spiders == 2
    >>> assert f.wires == g.wires == [0, 0, 1, 1]

    >>> assert (f >> g).n_spiders == 3
    >>> assert (f >> g).wires == [0, 0, 1, 1, 2, 2]

    >>> assert (f @ g).n_spiders == 4
    >>> assert (f @ g).wires == [0, 1, 0, 2, 1, 3, 2, 3]
    """
    def __init__(
            self, dom: Ty, cod: Ty, boxes: tuple[Box, ...],
            wires: tuple[Any, ...], spider_types: Mapping[Any, Ty] = None):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        self.dom, self.cod, self.boxes = dom, cod, boxes
        for box in boxes:
            assert_isinstance(box, self.category.ar)
        if len(wires) != len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod):
            raise ValueError
        if spider_types is None:
            port_types = list(self.dom) + sum(
                [list(box.dom @ box.cod) for box in boxes], [])\
                + list(self.cod)
            spider_types = {}
            for spider, typ in zip(wires, port_types):
                adjoint = getattr(typ, 'r', typ)
                if spider in spider_types:
                    if spider_types[spider] not in [typ, adjoint]:
                        raise AxiomError(messages.TYPE_ERROR.format(
                            typ, spider_types[spider]))
                else:
                    spider_types[spider] = typ
            spider_types = [spider_types[i] for i in sorted(spider_types)]
        relabeling = list(sorted(set(wires), key=lambda i: wires.index(i)))
        wires = [relabeling.index(spider) for spider in wires]
        spider_types = {i: t for i, t in enumerate(spider_types)}\
            if isinstance(spider_types, list) else spider_types
        relabeling += list(sorted(set(spider_types) - set(relabeling)))
        spider_types = [spider_types[spider] for spider in relabeling]
        self.wires, self.spider_types = wires, spider_types

    @property
    def box_wires(self):
        """
        The wires connecting the boxes of a hypergraph diagram.

        Returns a list of length :code:`len(self.boxes)` such that::

            dom_wires, cod_wires = self.box_wires[i]
            len(dom_wires) == len(box.dom) and len(cod_wires) == len(box.cod)

        for :code:`box = self.boxes[i]`.
        """
        result, i = [], len(self.dom)
        for box in self.boxes:
            dom_wires = self.wires[i:i + len(box.dom)]
            cod_wires = self.wires[i + len(box.dom):i + len(box.dom @ box.cod)]
            result.append((dom_wires, cod_wires))
            i += len(box.dom @ box.cod)
        return result

    @property
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
        Node('input', i=0, obj=frobenius.Ob('x'))
        Node('dom', depth=0, i=0, obj=frobenius.Ob('x'))
        Node('cod', depth=0, i=0, obj=frobenius.Ob('y'))
        Node('cod', depth=0, i=1, obj=frobenius.Ob('y'))
        Node('dom', depth=1, i=0, obj=frobenius.Ob('y'))
        Node('dom', depth=1, i=1, obj=frobenius.Ob('y'))
        Node('cod', depth=1, i=0, obj=frobenius.Ob('z'))
        Node('output', i=0, obj=frobenius.Ob('z'))
        """
        inputs = [Node("input", i=i, obj=obj)
                  for i, obj in enumerate(self.dom.inside)]
        doms_and_cods = sum([[
            Node(kind, depth=depth, i=i, obj=obj)
            for i, obj in enumerate(typ.inside)]
            for depth, box in enumerate(self.boxes)
            for kind, typ in [("dom", box.dom), ("cod", box.cod)]], [])
        outputs = [Node("output", i=i, obj=obj)
                   for i, obj in enumerate(self.cod.inside)]
        return inputs + doms_and_cods + outputs

    @property
    def n_spiders(self):
        """ The number of spiders in a hypergraph diagram. """
        return len(self.spider_types)

    @property
    def scalar_spiders(self):
        """ The zero-legged spiders in a hypergraph diagram. """
        return [i for i in range(self.n_spiders) if not self.wires.count(i)]

    @classmethod
    def id(cls, dom=None) -> Hypergraph:
        dom = cls.category.ob() if dom is None else dom
        return cls(dom, dom, [], 2 * list(range(len(dom))))

    twist = id

    @mmap
    def then(self, other):
        """
        Composition of two hypergraph diagrams, i.e. their :func:`pushout`.
        """
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod, boxes = self.dom, other.cod, self.boxes + other.boxes
        self_boundary = self.wires[len(self.wires) - len(self.cod):]
        other_boundary = other.wires[:len(other.dom)]
        self_pushout, other_pushout = pushout(
            self.n_spiders, other.n_spiders, self_boundary, other_boundary)
        wires = [
            self_pushout[i] for i in self.wires[
                :len(self.wires) - len(self.cod)]]\
            + [other_pushout[i] for i in other.wires[len(other.dom):]]
        spider_types = {
            self_pushout[i]: t for i, t in enumerate(self.spider_types)}
        spider_types.update({
            other_pushout[i]: t for i, t in enumerate(other.spider_types)})
        return type(self)(dom, cod, boxes, wires, spider_types)

    @mmap
    def tensor(self, other):
        """ Tensor of two hypergraph diagrams, i.e. their disjoint union. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        dom_wires = self.wires[:len(self.dom)] + [
            self.n_spiders + i for i in other.wires[:len(other.dom)]]
        box_wires = self.wires[len(self.dom):-len(self.cod) or len(self.wires)]
        box_wires += [self.n_spiders + i for i in other.wires[
            len(other.dom):-len(other.cod) or len(other.wires)]]
        cod_wires = self.wires[len(self.wires) - len(self.cod):] + [
            self.n_spiders + i
            for i in other.wires[len(other.wires) - len(other.cod):]]
        wires = dom_wires + box_wires + cod_wires
        spiders = self.spider_types + other.spider_types
        return type(self)(dom, cod, boxes, wires, spiders)

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
        boxes = [box.dagger() for box in self.boxes[::-1]]
        dom_wires = self.wires[len(self.wires) - len(self.cod):]
        box_wires = sum([
            cod_wires + dom_wires
            for dom_wires, cod_wires in self.box_wires[::-1]], [])
        cod_wires = self.wires[:len(self.dom)]
        wires = dom_wires + box_wires + cod_wires
        return type(self)(dom, cod, boxes, wires, self.spider_types)

    @classmethod
    def swap(cls, left, right):
        dom, cod = left @ right, right @ left
        boxes, wires = [], list(range(len(dom)))\
            + list(range(len(left), len(dom))) + list(range(len(left)))
        return cls(dom, cod, boxes, wires)

    braid = swap

    @classmethod
    def spiders(cls, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, spider_types = [], list(typ)
        wires = (n_legs_in + n_legs_out) * list(range(len(typ)))
        return cls(dom, cod, boxes, wires, spider_types)

    cup_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.ar.cup_factory(left, right)))
    cap_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.ar.cap_factory(left, right)))

    @classmethod
    def cups(cls, left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        return cls(left @ right, cls.category.ob(), [], wires)

    @classmethod
    def caps(cls, left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        return cls(cls.category.ob(), left @ right, [], wires)

    def transpose(self, left=False):
        """ The transpose of a hypergraph diagram. """
        return self.category.ar.transpose(self, left)

    @classmethod
    def trace_factory(cls, arg: Hypergraph, left=False):
        """
        The trace of one wire in a hypergraph,
        called by :meth:`make_progressive`.

        Parameters:
            left : Whether to trace on the left or right.
        """
        return cls.category.ar.trace_factory.__func__(cls, arg, left)

    def trace(self, n=1, left=False):
        """
        The trace of a hypergraph is its pre- and post-composition with
        cups and caps to form a feedback loop.

        Parameters:
            diagram : The diagram to trace.
            left : Whether to trace on the left or right.
        """
        return self.category.ar.trace(self, n, left)

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
        boxes, box_wires = list(self.boxes), list(self.box_wires)
        boxes[i], boxes[j] = boxes[j], boxes[i]
        box_wires[i], box_wires[j] = box_wires[j], box_wires[i]
        dom_wires = self.wires[:len(self.dom)]
        cod_wires = self.wires[len(self.wires) - len(self.cod):]
        wires = dom_wires + sum([c + d for c, d in box_wires], []) + cod_wires
        return type(self)(self.dom, self.cod, boxes, wires, self.spider_types)

    def simplify(self):
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
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes)):
                result = self.interchange(i, j)
                if len(result.to_diagram()) < len(self.to_diagram()):
                    return result.simplify()
        return self

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Hypergraph):
            return False
        return self.is_parallel(other) and is_isomorphic(
            self.to_graph(), other.to_graph(), lambda x, y: x == y)

    def __hash__(self):
        return hash(getattr(self, attr) for attr in [
            'dom', 'cod', 'boxes', 'wires', 'n_spiders'])

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
    def is_monogamous(self):
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
        """
        inputs = self.wires[:len(self.dom)]
        outputs = self.wires[len(self.wires) - len(self.cod):]
        for dom_wires, cod_wires in self.box_wires:
            inputs += cod_wires
            outputs += dom_wires
        return sorted(inputs) == sorted(outputs)\
            == list(range(self.n_spiders - len(self.scalar_spiders)))

    @property
    def is_bijective(self):
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
        return all(
            self.wires.count(i) in [0, 2] for i in range(self.n_spiders))

    @property
    def bijection(self):
        """
        Bijection between ports.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> list(zip(f.wires, f.bijection))
        [(0, 1), (0, 0), (1, 3), (1, 2)]
        """
        if not self.is_bijective:
            raise ValueError
        result = {}
        for source, spider in enumerate(self.wires):
            if spider in self.wires[source + 1:]:
                target = self.wires[source + 1:].index(spider) + source + 1
                result[source], result[target] = target, source
        return [result[source] for source in sorted(result)]

    @property
    def is_progressive(self):
        """
        Checks progressivity, i.e. wires are monotone w.r.t. box index.
        If the diagram is progressive, monogamous and it doesn't have any
        scalar spiders, then it actually lives in a symmetric monoidal
        category, i.e. it can be drawn using only swaps.

        Examples
        --------
        >>> from discopy.frobenius import Ty, Box, Hypergraph as H
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y).to_hypergraph()
        >>> assert f.is_progressive
        >>> assert (f >> f[::-1]).is_progressive

        >>> cycle = H.caps(x, x) >> x @ (f >> f[::-1]) >> H.cups(x, x)
        >>> assert not cycle.is_progressive

        >>> assert not H.cups(x, x).is_progressive
        """
        if not self.is_monogamous:
            return False
        scan = set(self.wires[:len(self.dom)])
        for dom_wires, cod_wires in self.box_wires:
            if not set(dom_wires) <= scan:
                return False
            scan = scan.union(set(cod_wires))
        return True

    def make_bijective(self):
        """
        Introduces :class:`Spider` boxes to make self bijective.

        Example
        -------
        >>> from discopy.frobenius import Ty, Spider, Hypergraph as H
        >>> spider = H.spiders(1, 2, Ty('x')).make_bijective()
        >>> assert spider.boxes == [Spider(3, 0, Ty('x'))]
        >>> assert spider.wires == [0, 0, 1, 2, 1, 2]
        """
        boxes, wires, spider_types =\
            self.boxes.copy(), self.wires.copy(), self.spider_types.copy()
        for i, typ in reversed(list(enumerate(self.spider_types))):
            ports = [port for port, spider in enumerate(wires) if spider == i]
            n_legs = len(ports)
            if n_legs not in [0, 2]:
                boxes.append(self.category.ar.spider_factory(n_legs, 0, typ))
                for j, port in enumerate(ports):
                    wires[port] = len(spider_types) + j
                new_wires =\
                    list(range(len(spider_types), len(spider_types) + n_legs))
                wires = wires[:len(wires) - len(self.cod)] + new_wires\
                    + wires[len(wires) - len(self.cod):]
                spider_types += n_legs * [typ]
                del spider_types[i]
                wires = [j - 1 if j > i else j for j in wires]
        return type(self)(self.dom, self.cod, boxes, wires, spider_types)

    def make_monogamous(self):
        """
        Introduce :class:`Cup` and :class:`Cap` boxes to make self monogamous.

        Example
        -------
        >>> from discopy.frobenius import Ty, Hypergraph as H, Cup, Cap, Spider
        >>> x = Ty('x')
        >>> assert H.caps(x, x).make_monogamous() == H.from_box(Cap(x, x))
        >>> assert H.cups(x, x).make_monogamous() == H.from_box(Cup(x, x))
        >>> spider = H.spiders(2, 1, x).make_monogamous()
        >>> assert spider.boxes == [Cap(x, x), Spider(3, 0, x)]
        >>> assert spider.wires == [0, 1, 2, 3, 0, 1, 2, 3]
        """
        if not self.is_bijective:
            return self.make_bijective().make_monogamous()
        if self.is_monogamous:
            return self
        dom, cod = self.dom, self.cod
        boxes, wires = list(self.boxes), list(self.wires)
        spider_types = dict(enumerate(self.spider_types))
        for kinds, cups_or_caps in [
                (["input", "cod"], "cups"),
                (["dom", "output"], "caps")]:
            for source, spider in [
                    (source, spider) for source, (spider, port)
                    in enumerate(zip(self.wires, self.ports))
                    if port.kind in kinds]:
                if spider not in wires[source + 1:]:
                    continue
                target = wires[source + 1:].index(spider) + source + 1
                typ = spider_types[spider]
                if self.ports[target].kind in kinds:
                    left, right = len(spider_types), len(spider_types) + 1
                    wires[source], wires[target] = left, right
                    if cups_or_caps == "cups":
                        boxes.append(self.category.ar.cup_factory(typ, typ))
                        wires = wires[:len(wires) - len(self.cod)]\
                            + [left, right]\
                            + wires[len(wires) - len(self.cod):]
                    else:
                        boxes = [
                            self.category.ar.cap_factory(typ, typ)] + boxes
                        wires = wires[:len(self.dom)] + [left, right]\
                            + wires[len(self.dom):]
                    spider_types[left] = spider_types[right] = typ
                    del spider_types[spider]
                    return type(self)(dom, cod, boxes, wires, spider_types)\
                        .make_monogamous()

    def make_progressive(self):
        """
        Calls :meth:`Hypergraph.trace_factory` boxes to make self progressive.

        Example
        -------
        >>> from discopy.frobenius import Ty, Box, Cup, Cap, Hypergraph as H
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x).to_hypergraph()
        >>> assert f.trace().make_progressive().boxes\\
        ...     == [Cap(x, x), f.boxes[0], Cup(x, x)]
        """
        if not self.is_monogamous:
            return self.make_monogamous().make_progressive()
        dom, cod = self.dom, self.cod
        boxes, wires = list(self.boxes), list(self.wires)
        spider_types = {i: x for i, x in enumerate(self.spider_types)}
        port = len(self.dom)
        for box in self.boxes:
            for j, typ in enumerate(box.dom):
                source = port + j
                spider, target = wires[source], self.bijection[source]
                if target > source:
                    input_spider, output_spider =\
                        range(len(spider_types), len(spider_types) + 2)
                    wires[source], wires[target] = input_spider, output_spider
                    wires = wires[:len(dom)] + [input_spider]\
                        + wires[len(dom):] + [output_spider]
                    dom, cod = dom @ typ, cod @ typ
                    spider_types.update({
                        input_spider: typ, output_spider: typ})
                    del spider_types[spider]
                    arg = type(self)(dom, cod, boxes, wires, spider_types)
                    return self.trace_factory(arg.make_progressive())
            port += len(box.dom @ box.cod)
        return self

    def to_diagram(self):
        """
        Downgrade to :class:`Diagram`, called by :code:`print`.

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
        diagram = self.make_progressive()
        graph = Graph()
        graph.add_nodes_from(diagram.ports)
        graph.add_edges_from([
            (diagram.ports[i], diagram.ports[j])
            for i, j in enumerate(diagram.bijection)])
        graph.add_nodes_from([
            Node("box", depth=depth, box=box)
            for depth, box in enumerate(diagram.boxes)])
        graph.add_nodes_from([
            Node("box", depth=len(diagram.boxes) + i,
                 box=self.category.ar.spiders(0, 0, diagram.spider_types[s]))
            for i, s in enumerate(diagram.scalar_spiders)])
        return drawing.nx2diagram(graph, self.category.ar)

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
        return old.functor_factory(
            ob=lambda typ: typ, ar=cls.from_box,
            cod=Category(cls.category.ob, cls))(old)

    def to_graph(self):
        """
        Translate a hypergraph into a labeled graph with nodes for inputs,
        outputs, boxes, domain, codomain and spiders.
        """
        graph = Graph()
        graph.add_nodes_from(
            Node("spider", i=i) for i in range(self.n_spiders))
        graph.add_nodes_from(
            [(Node("input", i=i), dict(i=i)) for i, _ in enumerate(self.dom)])
        graph.add_edges_from(
            (Node("input", i=i), Node("spider", i=j))
            for i, j in enumerate(self.wires[:len(self.dom)]))
        for i, (dom_wires, cod_wires) in enumerate(self.box_wires):
            box_node = Node("box", i=i)
            graph.add_node(box_node, box=self.boxes[i])
            for case, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    spider_node = Node("spider", i=spider)
                    port_node = Node(case, i=i, j=j)
                    graph.add_node(port_node, j=j)
                    graph.add_edge(box_node, port_node)
                    graph.add_edge(port_node, spider_node)
        graph.add_nodes_from(
            [(Node("output", i=i), dict(i=i)) for i, _ in enumerate(self.cod)])
        graph.add_edges_from(
            (Node("output", i=i), Node("spider", i=j))
            for i, j in enumerate(
                self.wires[len(self.wires) - len(self.cod):]))
        return graph

    def spring_layout(self, seed=None, k=None):
        """ Computes a layout using a force-directed algorithm. """
        if seed is not None:
            random.seed(seed)
        graph, pos = self.to_graph(), {}
        height = len(self.boxes) + self.n_spiders
        width = max(len(self.dom), len(self.cod))
        for i, _ in enumerate(self.dom):
            pos[Node("input", i=i)] = (i, height)
        for i, (dom_wires, cod_wires) in enumerate(self.box_wires):
            box_node = Node("box", i=i)
            pos[box_node] = (
                random.uniform(-width / 2, width / 2),
                random.uniform(0, height))
            for kind, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    pos[Node(kind, i=i, j=j)] = pos[box_node]
        for i in range(self.n_spiders):
            pos[Node("spider", i=i)] = (
                random.uniform(-width / 2, width / 2),
                random.uniform(0, height))
        for i, _ in enumerate(self.cod):
            pos[Node("output", i=i)] = (i, 0)
        fixed = [Node("input", i=i) for i, _ in enumerate(self.dom)] + [
            Node("output", i=i) for i, _ in enumerate(self.cod)] or None
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
        for i, (dom_wires, cod_wires) in enumerate(self.box_wires):
            box_node = Node("box", i=i)
            for kind, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    port_node = Node(kind, i=i, j=j)
                    x, y = pos[box_node]
                    if not getattr(self.boxes[i], "draw_as_spider", False):
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
        Node('input', i=0, obj=frobenius.Ob('x'))
        Node('dom', depth=0, i=0, obj=frobenius.Ob('x'))
        Node('cod', depth=0, i=0, obj=frobenius.Ob('y'))
        Node('cod', depth=0, i=1, obj=frobenius.Ob('z'))
        Node('output', i=0, obj=frobenius.Ob('y'))
        Node('output', i=1, obj=frobenius.Ob('z'))
        """
        spider_types = list(box.dom @ box.cod)
        wires = 2 * list(range(len(box.dom)))\
            + 2 * list(range(len(box.dom), len(box.dom @ box.cod)))
        return cls(box.dom, box.cod, [box], wires, spider_types)
