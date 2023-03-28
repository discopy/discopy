# -*- coding: utf-8 -*-

"""
The free hypergraph category with diagrams encoded as cospans of hypergraphs.


Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        pushout

Axioms
------

**Spiders**

We can check spider fusion, i.e. special commutative Frobenius algebra.

>>> x, y, z = map(Ty, "xyz")
>>> split, merge = spiders(1, 2, x), spiders(2, 1, x)
>>> unit, counit = spiders(0, 1, x), spiders(1, 0, x)

* (Co)commutative (co)monoid:

>>> assert unit @ Id(x) >> merge == Id(x) == Id(x) @ unit >> merge
>>> assert merge @ Id(x) >> merge == Id(x) @ merge >> merge
>>> assert Swap(x, x) >> merge == merge
>>> assert split >> counit @ Id(x) == Id(x) == split >> Id(x) @ counit
>>> assert split >> split @ Id(x) == split >> Id(x) @ split
>>> assert split >> Swap(x, x) == split

* Frobenius:

>>> assert split @ Id(x) >> Id(x) @ merge\\
...     == merge >> split\\
...     == Id(x) @ split >> merge @ Id(x)\\
...     == spiders(2, 2, x)

* Speciality:

>>> assert split >> merge == spiders(1, 1, x) == Id(x)

* Coherence:

>>> assert spiders(0, 1, x @ x) == unit @ unit
>>> assert spiders(2, 1, x @ x) == Id(x) @ Swap(x, x) @ Id(x) >> merge @ merge
>>> assert spiders(1, 0, x @ x) == counit @ counit
>>> assert spiders(1, 2, x @ x) == split @ split >> Id(x) @ Swap(x, x) @ Id(x)

**Snakes**

Special commutative Frobenius algebras imply compact-closedness, i.e.

* Snake equations:

>>> left_snake = lambda x: caps(x, x.r) @ Id(x) >> Id(x) @ cups(x.r, x)
>>> right_snake = lambda x: Id(x) @ caps(x.r, x) >> cups(x, x.r) @ Id(x)
>>> assert left_snake(x) == Id(x) == right_snake(x)
>>> assert left_snake(x @ y) == Id(x @ y) == right_snake(x @ y)

* Yanking (a.k.a. Reidemeister move 1):

>>> right_loop = lambda x: Id(x) @ caps(x, x.r)\\
...     >> Swap(x, x) @ Id(x.r) >> Id(x) @ cups(x, x.r)
>>> left_loop = lambda x: caps(x.r, x) @ Id(x)\\
...     >> Id(x.r) @ Swap(x, x) >> cups(x.r, x) @ Id(x)
>>> top_loop = lambda x: caps(x, x.r) >> Swap(x, x.r)
>>> bottom_loop = lambda x: Swap(x, x.r) >> cups(x.r, x)
>>> reidemeister1 = lambda x:\\
...     top_loop(x) == caps(x.r, x) and bottom_loop(x) == cups(x, x.r)\\
...     and left_loop(x) == Id(x) == right_loop(x)
>>> assert reidemeister1(x) and reidemeister1(x @ y) and reidemeister1(Ty())

* Coherence:

>>> assert caps(x @ y, y @ x)\\
...     == caps(x, x) @ caps(y, y) >> Id(x) @ Swap(x, y @ y)\\
...     == spiders(0, 2, x @ y) >> Id(x @ y) @ Swap(x, y)
>>> assert caps(x, x) >> cups(x, x) == spiders(0, 0, x)

**Swaps**

We can also check that the axioms for symmetry hold on the nose.

* Involution (a.k.a. Reidemeister move 2):

>>> reidermeister2 = lambda x, y: Swap(x, y) >> Swap(y, x) == Id(x @ y)
>>> assert reidermeister2(x, y) and reidermeister2(x @ y, z)

* Yang-Baxter (a.k.a. Reidemeister move 3):

>>> left = Swap(x, y) @ Id(z)\\
...     >> Id(y) @ Swap(x, z)\\
...     >> Swap(y, z) @ Id(x)
>>> right = Id(x) @ Swap(y, z)\\
...     >> Swap(x, z) @ Id(y)\\
...     >> Id(z) @ Swap(x, y)
>>> assert left == right

* Coherence (a.k.a. pentagon equations):

>>> assert Swap(x, y @ z) == Swap(x, y) @ Id(z) >> Id(y) @ Swap(x, z)
>>> assert Swap(x @ y, z) == Id(x) @ Swap(y, z) >> Swap(x, z) @ Id(y)

* Naturality:

>>> f = Box("f", x, y)
>>> assert f @ Id(z) >> Swap(f.cod, z) == Swap(f.dom, z) >> Id(z) @ f
>>> assert Id(z) @ f >> Swap(z, f.cod) == Swap(z, f.dom) >> f @ Id(z)
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from networkx import Graph, connected_components, spring_layout, draw_networkx

from discopy import cat, monoidal, drawing, frobenius
from discopy.braided import BinaryBoxConstructor
from discopy.cat import AxiomError, Composable
from discopy.drawing import Node
from discopy.frobenius import Ty, Category
from discopy.monoidal import Whiskerable, assert_isatomic
from discopy.utils import (
    factory_name,
    assert_isinstance,
)

Pushout = tuple[dict[int, int], dict[int, int]]


def pushout(
        left: int, right: int,
        left_boundary: list[int], right_boundary: list[int]) -> Pushout:
    """
    Computes the pushout of two finite mappings using connected components.

    Parameters:
        left : The size of the left set.
        right : The size of the right set.
        left_boundary : The mapping from boundary to left.
        right_boundary : The mapping from boundary to right.

    Examples
    --------
    >>> assert pushout(2, 3, [1], [0]) == ({0: 0, 1: 1}, {0: 1, 1: 2, 2: 3})
    """
    if len(left_boundary) != len(right_boundary):
        raise ValueError
    components, left_pushout, right_pushout = set(), dict(), dict()
    left_proper = sorted(set(range(left)) - set(left_boundary))
    left_pushout.update({j: i for i, j in enumerate(left_proper)})
    graph = Graph([
        (("middle", i), ("left", j)) for i, j in enumerate(left_boundary)] + [
        (("middle", i), ("right", j)) for i, j in enumerate(right_boundary)])
    for i, component in enumerate(connected_components(graph)):
        components.add(i)
        for case, j in component:
            if case == "left":
                left_pushout[j] = len(left_proper) + i
            if case == "right":
                right_pushout[j] = len(left_proper) + i
    right_proper = set(range(right)) - set(right_boundary)
    right_pushout.update({
        j: len(left_proper) + len(components) + i
        for i, j in enumerate(right_proper)})
    return left_pushout, right_pushout


class Diagram(Composable[Ty], Whiskerable):
    """
    Diagram in a hypergraph category.

    Parameters:
        dom (frobenius.Ty) : The domain of the diagram, i.e. its input.
        cod (frobenius.Ty) : The codomain of the diagram, i.e. its output.
        boxes (tuple[Box, ...]) : The boxes inside the diagram.
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
    >>> x, y, z = map(Ty, "xyz")

    >>> assert Id(x @ y @ z).n_spiders == 3
    >>> assert Id(x @ y @ z).wires == [0, 1, 2, 0, 1, 2]

    >>> assert Swap(x, y).n_spiders == 2
    >>> assert Swap(x, y).wires == [0, 1, 1, 0]

    >>> assert spiders(1, 2, x @ y).n_spiders == 2
    >>> assert spiders(1, 2, x @ y).wires == [0, 1, 0, 1, 0, 1]
    >>> assert spiders(0, 0, x @ y @ z).n_spiders == 3
    >>> assert spiders(0, 0, x @ y @ z).wires == []

    >>> f, g = Box('f', x, y), Box('g', y, z)

    >>> assert f.n_spiders == g.n_spiders == 2
    >>> assert f.wires == g.wires == [0, 0, 1, 1]

    >>> assert (f >> g).n_spiders == 3
    >>> assert (f >> g).wires == [0, 0, 1, 1, 2, 2]

    >>> assert (f @ g).n_spiders == 4
    >>> assert (f @ g).wires == [0, 1, 0, 2, 1, 3, 2, 3]
    """
    def __init__(
            self, dom: frobenius.Ty, cod: frobenius.Ty, boxes: tuple[Box, ...],
            wires: tuple[Any, ...], spider_types: Mapping[Any, Ty] = None):
        assert_isinstance(dom, Ty)
        assert_isinstance(cod, Ty)
        self.dom, self.cod, self.boxes = dom, cod, boxes
        for box in boxes:
            assert_isinstance(box, Box)
        if len(wires) != len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod):
            raise ValueError
        if spider_types is None:
            port_types = list(self.dom) + sum(
                [list(box.dom @ box.cod) for box in boxes], [])\
                + list(self.cod)
            spider_types = {}
            for spider, typ in zip(wires, port_types):
                if spider in spider_types:
                    if spider_types[spider] != typ:
                        raise AxiomError
                else:
                    spider_types[spider] = typ
            spider_types = [spider_types[i] for i in sorted(spider_types)]
        relabeling = list(sorted(set(wires), key=lambda i: wires.index(i)))
        wires = [relabeling.index(spider) for spider in wires]
        spider_types = {i: t for i, t in enumerate(spider_types)}\
            if isinstance(spider_types, list) else spider_types
        relabeling += list(sorted(set(spider_types) - set(relabeling)))
        spider_types = [spider_types[spider] for spider in relabeling]
        self._wires, self._spider_types = wires, spider_types

    @property
    def wires(self):
        """ The wires of a diagram, i.e. a map from ports to spiders. """
        return self._wires

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
        >>> x, y, z = map(Ty, "xyz")
        >>> f, g = Box('f', x, y @ y), Box('g', y @ y, z)
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
    def spider_types(self):
        """ List of types for each spider. """
        return self._spider_types

    @property
    def n_spiders(self):
        """ The number of spiders in a hypergraph diagram. """
        return len(self.spider_types)

    @property
    def scalar_spiders(self):
        """ The zero-legged spiders in a hypergraph diagram. """
        return [i for i in range(self.n_spiders) if not self.wires.count(i)]

    @staticmethod
    def id(dom=Ty()) -> Diagram:
        return Diagram(dom, dom, [], 2 * list(range(len(dom))))

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
        return Diagram(dom, cod, boxes, wires, spider_types)

    def tensor(self, other=None, *rest):
        """ Tensor of two hypergraph diagrams, i.e. their disjoint union. """
        if other is None or rest:
            return monoidal.Diagram.tensor(self, other, *rest)
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
        return Diagram(dom, cod, boxes, wires, spiders)

    def dagger(self):
        """
        Dagger of a hypergraph diagram, called with :code:`[::-1]`.

        Examples
        --------
        >>> x, y, z = map(Ty, "xyz")
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert (f >> g)[::-1] == g[::-1] >> f[::-1]
        >>> assert spiders(1, 2, x @ y)[::-1] == spiders(2, 1, x @ y)
        """
        dom, cod = self.cod, self.dom
        boxes = [box.dagger() for box in self.boxes[::-1]]
        dom_wires = self.wires[len(self.wires) - len(self.cod):]
        box_wires = sum([
            cod_wires + dom_wires
            for dom_wires, cod_wires in self.box_wires[::-1]], [])
        cod_wires = self.wires[:len(self.dom)]
        wires = dom_wires + box_wires + cod_wires
        return Diagram(dom, cod, boxes, wires, self.spider_types)

    @staticmethod
    def swap(left, right):
        dom, cod = left @ right, right @ left
        boxes, wires = [], list(range(len(dom)))\
            + list(range(len(left), len(dom))) + list(range(len(left)))
        return Diagram(dom, cod, boxes, wires)

    @staticmethod
    def spiders(n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, spider_types = [], list(typ)
        wires = (n_legs_in + n_legs_out) * list(range(len(typ)))
        return Diagram(dom, cod, boxes, wires, spider_types)

    @staticmethod
    def cups(left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        return Diagram(left @ right, Ty(), [], wires)

    @staticmethod
    def caps(left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        return Diagram(Ty(), left @ right, [], wires)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['dom', 'cod', 'boxes', 'wires', 'n_spiders'])

    def __repr__(self):
        spider_types = f", spider_types={self.spider_types}"\
            if self.scalar_spiders else ""
        return factory_name(type(self))\
            + f"(dom={repr(self.dom)}, cod={repr(self.cod)}, " \
              f"boxes={repr(self.boxes)}, " \
              f"wires={repr(self.wires)}{spider_types})"

    def __str__(self):
        return str(self.downgrade())

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

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> assert f.is_monogamous
        >>> assert (f >> f[::-1]).is_monogamous

        >>> assert spiders(0, 0, x).is_monogamous

        >>> cycle = caps(x, x) >> Id(x) @ (f >> f[::-1]) >> cups(x, x)
        >>> assert cycle.is_monogamous

        >>> assert not f.transpose().is_monogamous
        >>> assert not cups(x, x).is_monogamous
        >>> assert not spiders(1, 2, x).is_monogamous
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

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> assert f.is_bijective and f.transpose().is_bijective
        >>> assert cups(x, x).is_bijective and caps(x, x).is_bijective
        >>> assert spiders(0, 0, x).is_bijective
        >>> assert not spiders(1, 2, x).is_bijective
        """
        return all(
            self.wires.count(i) in [0, 2] for i in range(self.n_spiders))

    @property
    def bijection(self):
        """
        Bijection between ports.

        Examples
        --------

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
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

        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> assert f.is_progressive
        >>> assert (f >> f[::-1]).is_progressive

        >>> cycle = caps(x, x) >> Id(x) @ (f >> f[::-1]) >> cups(x, x)
        >>> assert not cycle.is_progressive

        >>> assert not cups(x, x).is_progressive
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

        >>> spider = spiders(1, 2, Ty('x')).make_bijective()
        >>> assert spider.boxes == [Spider(3, 0, Ty('x'))]
        >>> assert spider.wires == [0, 0, 1, 2, 1, 2]
        """
        boxes, wires, spider_types =\
            self.boxes.copy(), self.wires.copy(), self.spider_types.copy()
        for i, typ in reversed(list(enumerate(self.spider_types))):
            ports = [port for port, spider in enumerate(wires) if spider == i]
            n_legs = len(ports)
            if n_legs not in [0, 2]:
                boxes.append(Spider(n_legs, 0, typ))
                for j, port in enumerate(ports):
                    wires[port] = len(spider_types) + j
                new_wires =\
                    list(range(len(spider_types), len(spider_types) + n_legs))
                wires = wires[:len(wires) - len(self.cod)] + new_wires\
                    + wires[len(wires) - len(self.cod):]
                spider_types += n_legs * [typ]
                del spider_types[i]
                wires = [j - 1 if j > i else j for j in wires]
        return Diagram(self.dom, self.cod, boxes, wires, spider_types)

    def make_monogamous(self):
        """
        Introduce :class:`Cup` and :class:`Cap` boxes to make self monogamous.

        >>> x = Ty('x')
        >>> assert caps(x, x).make_monogamous() == Cap(x, x)
        >>> assert cups(x, x).make_monogamous() == Cup(x, x)
        >>> spider = spiders(2, 1, x).make_monogamous()
        >>> assert spider.boxes == [Cap(x, x), Spider(3, 0, x)]
        >>> assert spider.wires == [0, 1, 2, 3, 0, 1, 2, 3]
        """
        diagram = self if self.is_bijective else self.make_bijective()
        if diagram.is_monogamous:
            return diagram
        dom, cod = diagram.dom, diagram.cod
        boxes, wires = list(diagram.boxes), list(diagram.wires)
        spider_types = dict(enumerate(diagram.spider_types))
        for kinds, box_cls in [
                (["input", "cod"], Cup),
                (["dom", "output"], Cap)]:
            for source, spider in [
                    (source, spider) for source, (spider, port)
                    in enumerate(zip(diagram.wires, diagram.ports))
                    if port.kind in kinds]:
                if spider not in wires[source + 1:]:
                    continue
                target = wires[source + 1:].index(spider) + source + 1
                typ = spider_types[spider]
                if diagram.ports[target].kind in kinds:
                    left, right = len(spider_types), len(spider_types) + 1
                    wires[source], wires[target] = left, right
                    if box_cls == Cup:
                        boxes.append(box_cls(typ, typ))
                        wires = wires[:len(wires) - len(diagram.cod)]\
                            + [left, right]\
                            + wires[len(wires) - len(diagram.cod):]
                    else:
                        boxes = [box_cls(typ, typ)] + boxes
                        wires = wires[:len(diagram.dom)] + [left, right]\
                            + wires[len(diagram.dom):]
                    spider_types[left] = spider_types[right] = typ
                    del spider_types[spider]
                    return Diagram(dom, cod, boxes, wires, spider_types)\
                        .make_monogamous()

    def make_progressive(self):
        """
        Introduce :class:`Cup` and :class:`Cap` boxes to make self progressive.

        Examples
        --------
        >>> trace = lambda d:\\
        ...     caps(d.dom, d.dom) >> Id(d.dom) @ d >> cups(d.dom, d.dom)
        >>> x = Ty('x')
        >>> f = Box('f', x, x)
        >>> diagram = trace(f).make_progressive()
        >>> assert diagram.boxes == [Cap(x, x), f, Cup(x, x)]
        >>> assert diagram.wires == [0, 1, 0, 2, 2, 1]

        >>> g = Box('g', x @ x, x @ x)
        >>> assert trace(g).make_progressive().boxes\\
        ...     == [Cap(x, x), Cap(x, x), g, Cup(x, x), Cup(x, x)]
        """
        diagram = self if self.is_monogamous else self.make_monogamous()
        if diagram.is_progressive:
            return diagram
        dom, cod = diagram.dom, diagram.cod
        boxes, wires = list(diagram.boxes), list(diagram.wires)
        spider_types = {i: x for i, x in enumerate(diagram.spider_types)}
        bijection = diagram.bijection
        port = len(diagram.dom)
        for box in diagram.boxes:
            for j, typ in enumerate(box.dom):
                source = port + j
                spider, target = wires[source], bijection[source]
                if target > source:
                    cup, cap = Cup(typ, typ), Cap(typ, typ)
                    boxes = [cap] + boxes + [cup]
                    top, middle, bottom =\
                        range(len(spider_types), len(spider_types) + 3)
                    wires[source], wires[target] = top, bottom
                    wires = wires[:len(dom)] + [top, middle]\
                        + wires[len(dom):len(wires) - len(cod)]\
                        + [bottom, middle] + wires[len(wires) - len(cod):]
                    spider_types.update({top: typ, middle: typ, bottom: typ})
                    del spider_types[spider]
                    return Diagram(dom, cod, boxes, wires, spider_types)\
                        .make_progressive()
            port += len(box.dom @ box.cod)

    def downgrade(self):
        """
        Downgrade to :class:`frobenius.Diagram`, called by :code:`print`.

        Examples
        --------
        >>> x = Ty('x')
        >>> v = Box('v', Ty(), x @ x)
        >>> print(v >> Swap(x, x) >> v[::-1])
        v >> Swap(x, x) >> v[::-1]
        >>> print(x @ Swap(x, x) >> v[::-1] @ x)
        x @ Swap(x, x) >> v[::-1] @ x
        """
        diagram = self.make_progressive()
        graph = Graph()
        graph.add_nodes_from(diagram.ports)
        graph.add_edges_from([
            (diagram.ports[i], diagram.ports[j])
            for i, j in enumerate(diagram.bijection)])
        graph.add_nodes_from([
            Node("box", depth=depth, box=box.downgrade())
            for depth, box in enumerate(diagram.boxes)])
        graph.add_nodes_from([
            Node("box", depth=len(diagram.boxes) + i,
                 box=frobenius.Spider(0, 0, diagram.spider_types[s]))
            for i, s in enumerate(diagram.scalar_spiders)])
        return drawing.nx2diagram(graph, frobenius.Diagram)

    @staticmethod
    def upgrade(old: frobenius.Diagram) -> Diagram:
        """
        Turn a :class:`frobenius.Diagram` into a :class:`hypergraph.Diagram`.

        >>> x, y = map(Ty, "xy")
            >>> back_n_forth = lambda d: Diagram.upgrade(d.downgrade())
        >>> for d in [spiders(0, 0, x),
        ...           spiders(2, 3, x),
        ...           spiders(1, 2, x @ y)]:
        ...     assert back_n_forth(d) == d
        """
        return frobenius.Functor(
            ob=lambda typ: Ty(typ.name),
            ar=lambda box: Box(box.name, box.dom, box.cod),
            cod=Category(Ty, Diagram))(old)

    def spring_layout(self, seed=None, k=None):
        """ Computes planar position using a force-directed layout. """
        if seed is not None:
            random.seed(seed)
        height = len(self.boxes) + self.n_spiders
        width = max(len(self.dom), len(self.cod))
        graph, pos = Graph(), {}
        graph.add_nodes_from(
            Node("spider", i=i) for i in range(self.n_spiders))
        graph.add_edges_from(
            (Node("input", i=i), Node("spider", i=j))
            for i, j in enumerate(self.wires[:len(self.dom)]))
        for i, (dom_wires, cod_wires) in enumerate(self.box_wires):
            box_node = Node("box", i=i)
            graph.add_node(box_node)
            for case, wires in [("dom", dom_wires), ("cod", cod_wires)]:
                for j, spider in enumerate(wires):
                    spider_node = Node("spider", i=spider)
                    port_node = Node(case, i=i, j=j)
                    graph.add_edge(box_node, port_node)
                    graph.add_edge(port_node, spider_node)
        graph.add_edges_from(
            (Node("output", i=i), Node("spider", i=j))
            for i, j in enumerate(
                self.wires[len(self.wires) - len(self.cod):]))
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
        Draw a hypegraph diagram.

        Examples
        --------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ z)
        >>> f.draw(
        ...     path='docs/_static/hypergraph/box.png', seed=42)

        .. image:: /_static/hypergraph/box.png
            :align: center

        >>> (Spider(2, 2, x) >> f @ Id(x)).draw(
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
                    if not isinstance(self.boxes[i], Spider):
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

    def transpose(self, left=False):
        """ The transpose of a hypergraph diagram. """
        return frobenius.Diagram.transpose(self)


class Box(Diagram):
    """ Box in a :class:`discopy.hypergraph.Diagram`. """
    def __init__(self, name, dom, cod, is_dagger=False, data=None):
        self.name, self.dom, self.cod = name, dom, cod
        self.is_dagger, self.data = is_dagger, data
        boxes, spider_types = [self], list(dom @ cod)
        wires = 2 * list(range(len(dom)))\
            + 2 * list(range(len(dom), len(dom @ cod)))
        Diagram.__init__(self, dom, cod, boxes, wires, spider_types)

    def __eq__(self, other):
        if isinstance(other, Box):
            return cat.Box.__eq__(self, other)
        return Diagram.__eq__(self, other)

    def dagger(self):
        return Box(
            self.name, self.cod, self.dom, not self.is_dagger, self.data)

    def downgrade(self):
        return frobenius.Box(
            self.name, self.dom, self.cod,
            is_dagger=self.is_dagger, data=self.data)

    __repr__, __str__ = cat.Box.__repr__, cat.Box.__str__


class Cup(BinaryBoxConstructor, Box):
    """
    A box introduced by :meth:`Diagram.make_monogamous` and
    :meth:`Diagram.make_progressive`.

    Parameters:
        left : The atomic type.
        right : Its adjoint.
    """
    def __init__(self, left, right):
        assert_isatomic(left, Ty)
        assert_isatomic(right, Ty)
        BinaryBoxConstructor.__init__(self, left, right)
        name = f"Cup({left}, {right})"
        Box.__init__(self, name, left @ right, Ty())

    def dagger(self):
        return Cap(self.left, self.right)

    def downgrade(self):
        return frobenius.Cup(self.left, self.right)


class Cap(BinaryBoxConstructor, Box):
    """
    A box introduced by :meth:`Diagram.make_monogamous` and
    :meth:`Diagram.make_progressive`.

    Parameters:
        left : The atomic type.
        right : Its adjoint.
    """
    def __init__(self, left, right):
        assert_isatomic(left, Ty)
        assert_isatomic(right, Ty)
        BinaryBoxConstructor.__init__(self, left, right)
        name = f"Cap({left}, {right})"
        Box.__init__(self, name, Ty(), left @ right)

    def dagger(self):
        return Cup(self.left, self.right)

    def downgrade(self):
        return frobenius.Cap(self.left, self.right)


class Spider(Box):
    """
    A box introduced by :meth:`Diagram.make_bijective`.

    Parameters:
        n_legs_in : The number of legs in.
        n_legs_out : The number of legs out.
        typ : The type of the spider.

    Examples
    --------
    >>> x = Ty('x')
    >>> spider = Spider(1, 2, x)
    >>> assert spider.dom == x and spider.cod == x @ x
    """
    def __init__(self, n_legs_in: int, n_legs_out: int, typ: Ty):
        assert_isatomic(typ, Ty)
        self.typ = typ
        name = f"Spider({n_legs_in}, {n_legs_out}, {typ})"
        Box.__init__(self, name, typ ** n_legs_in, typ ** n_legs_out)

    def downgrade(self):
        return frobenius.Spider(len(self.dom), len(self.cod), self.typ)

    dagger = frobenius.Spider.dagger
    __repr__ = frobenius.Spider.__repr__


spiders, cups, caps = Diagram.spiders, Diagram.cups, Diagram.caps
Id, Swap = Diagram.id, Diagram.swap
