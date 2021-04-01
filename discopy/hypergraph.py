# -*- coding: utf-8 -*-

"""
Hypergraph categories.

Note
----

**Spiders**

We can check spider fusion, i.e. special commutative Frobenius algebra.

>>> x, y, z = types("x y z")
>>> split, merge = Spider(1, 2, x), Spider(2, 1, x)
>>> unit, counit = Spider(0, 1, x), Spider(1, 0, x)

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
...     == Spider(2, 2, x)

* Speciality:

>>> assert split >> merge == Spider(1, 1, x) == Id(x)

* Coherence:

>>> assert Spider(0, 1, x @ x) == unit @ unit
>>> assert Spider(2, 1, x @ x) == Id(x) @ Swap(x, x) @ Id(x) >> merge @ merge
>>> assert Spider(1, 0, x @ x) == counit @ counit
>>> assert Spider(1, 2, x @ x) == split @ split >> Id(x) @ Swap(x, x) @ Id(x)

**Snakes**

Special commutative Frobenius algebras imply compact-closedness, i.e.

* Snake equations:

>>> left_snake = lambda x: Cap(x, x.r) @ Id(x) >> Id(x) @ Cup(x.r, x)
>>> right_snake = lambda x: Id(x) @ Cap(x.r, x) >> Cup(x, x.r) @ Id(x)
>>> assert left_snake(x) == Id(x) == right_snake(x)
>>> assert left_snake(x @ y) == Id(x @ y) == right_snake(x @ y)

* Yanking (a.k.a. Reidemeister move 1):

>>> right_loop = lambda x: Id(x) @ Cap(x, x.r)\\
...     >> Swap(x, x) @ Id(x.r) >> Id(x) @ Cup(x, x.r)
>>> left_loop = lambda x: Cap(x.r, x) @ Id(x)\\
...     >> Id(x.r) @ Swap(x, x) >> Cup(x.r, x) @ Id(x)
>>> top_loop = lambda x: Cap(x, x.r) >> Swap(x, x.r)
>>> bottom_loop = lambda x: Swap(x, x.r) >> Cup(x.r, x)
>>> reidemeister1 = lambda x:\\
...     top_loop(x) == Cap(x.r, x) and bottom_loop(x) == Cup(x, x.r)\\
...     and left_loop(x) == Id(x) == right_loop(x)
>>> assert reidemeister1(x) and reidemeister1(x @ y) and reidemeister1(Ty())

* Coherence:

>>> assert Cap(x @ y, y @ x)\\
...     == Cap(x, x) @ Cap(y, y) >> Id(x) @ Swap(x, y @ y)\\
...     == Spider(0, 2, x @ y) >> Id(x @ y) @ Swap(x, y)
>>> assert Cap(x, x) >> Cup(x, x) == Spider(0, 0, x)

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

from networkx import Graph, connected_components

from discopy import cat, monoidal, rigid
from discopy.cat import AxiomError
from discopy.drawing import Node


class Ty(rigid.Ty):
    @staticmethod
    def upgrade(old):
        return Ty(*old.objects)

    @property
    def l(self):
        return Ty(*self.objects[::-1])

    r = l


def types(names):
    """ Transforms strings into lists of :class:`discopy.hypergraph.Ty`. """
    return map(Ty.upgrade, monoidal.types(names))



class Diagram(cat.Arrow):
    """
    Diagram in a hypergraph category.

    Parameters
    ----------

    dom : discopy.hypergraph.Ty
        Domain of the diagram.
    cod : discopy.hypergraph.Ty
        Codomain of the diagram.
    boxes : List[discopy.hypergraph.Box]
        List of :class:`discopy.symmetric.Box`.
    wires : List[int]
        List of wires from ports to spiders.
    n_spiders : int, optional
        Number of spiders, default is :code:`len(set(wires))`.

    Note
    ----

    The wires go from ports to spiders, they are given as a list of length::

        len(dom) + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod)

    The values themselves don't matter, they are simply labels for the spiders.
    We must have :code:`n_spiders >= len(set(wires))`, the spiders that don't
    appear as the target of a wire are scalar spiders, i.e. with zero legs.

    Abstractly, a hypergraph diagram can be seen as a cospan for the boundary::

        range(len(dom)) -> range(n_spiders) <- range(len(cod))

    together with a cospan for each box in boxes::

        range(len(box.dom)) -> range(n_spiders) <- range(len(box.cod))

    Composition of two hypergraph diagram is given by the pushout of the span::

        range(self.n_spiders) <- range(len(self.cod)) -> range(other.n_spiders)

    Examples
    --------

    >>> x, y, z = types("x y z")

    >>> assert Id(x @ y @ z).n_spiders == 3
    >>> assert Id(x @ y @ z).wires == [0, 1, 2, 0, 1, 2]

    >>> assert Swap(x, y).n_spiders == 2
    >>> assert Swap(x, y).wires == [0, 1, 1, 0]

    >>> assert Spider(1, 2, x @ y).n_spiders == 2
    >>> assert Spider(1, 2, x @ y).wires == [0, 1, 0, 1, 0, 1]
    >>> assert Spider(0, 0, x @ y @ z).n_spiders == 3
    >>> assert Spider(0, 0, x @ y @ z).wires == []

    >>> f, g = Box('f', x, y), Box('g', y, z)

    >>> assert f.n_spiders == g.n_spiders == 2
    >>> assert f.wires == g.wires == [0, 0, 1, 1]

    >>> assert (f >> g).n_spiders == 3
    >>> assert (f >> g).wires == [0, 0, 1, 1, 2, 2]

    >>> assert (f @ g).n_spiders == 4
    >>> assert (f @ g).wires == [0, 1, 0, 2, 1, 3, 2, 3]
    """
    def __init__(self, dom, cod, boxes, wires, n_spiders=None, _scan=True):
        super().__init__(dom, cod, boxes, _scan=False)
        if len(wires) != len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod):
            raise ValueError
        relabeling = sorted(set(wires), key=lambda i: wires.index(i))
        wires = [relabeling.index(i) for i in wires]
        n_connected_spiders = len(set(wires))
        if n_spiders is None:
            n_spiders = n_connected_spiders
        elif n_spiders < n_connected_spiders:
            raise ValueError
        self._n_spiders, self._wires = n_spiders, wires
        self._scalar_spiders = set(range(n_connected_spiders, n_spiders))
        if _scan:
            spider_types = {}
            for port, spider in zip(self.ports, wires):
                if spider in spider_types and spider_types[spider] != port.obj:
                    raise AxiomError
                elif spider not in spider_types:
                    spider_types[spider] = port.obj

    @property
    def scalar_spiders(self):
        """ The zero-legged spiders in a hypergraph diagram. """
        return self._scalar_spiders

    @property
    def n_spiders(self):
        """ The number of spiders in a hypergraph diagram. """
        return self._n_spiders

    @property
    def wires(self):
        """ The wires of a hypergraph diagram. """
        return self._wires

    @property
    def ports(self):
        """ The ports of a hypergraph diagram. """
        input_ports = [
            Node("input", i=i, obj=obj) for i, obj in enumerate(self.dom)]
        box_ports = sum([[
            Node("dom", i=i, obj=obj, depth=depth)
            for i, obj in enumerate(box.dom)] + [
            Node("cod", i=i, obj=obj, depth=depth)
            for i, obj in enumerate(box.cod)]
            for depth, box in enumerate(self.boxes)], [])
        output_ports = [
            Node("output", i=i, obj=obj) for i, obj in enumerate(self.cod)]
        return input_ports + box_ports + output_ports

    @property
    def is_hetero_monogamous(self):
        """
        Checks hetero-monogamy, i.e. whether self.wires induces a bijection::

            len(self.dom) + sum(len(box.dom) for box in boxes)
            == self.n_spiders - len(self._scalar_spiders)
            == sum(len(box.cod) for box in boxes) + len(self.dom)

        In that case, the diagram actually lives in a traced category.

        Examples
        --------

        >>> x, y = types(" x y")
        >>> f = Box('f', x, y)
        >>> assert f.is_hetero_monogamous
        >>> assert (f >> f[::-1]).is_hetero_monogamous

        >>> assert Spider(0, 0, x).is_hetero_monogamous

        >>> cycle = Cap(x, x) >> Id(x) @ (f >> f[::-1]) >> Cup(x, x)
        >>> assert cycle.is_hetero_monogamous

        >>> assert not f.transpose().is_hetero_monogamous
        >>> assert not Cup(x, x).is_hetero_monogamous
        >>> assert not Spider(1, 2, x).is_hetero_monogamous
        """
        inputs = self.wires[:len(self.dom)]
        outputs = self.wires[len(self.wires) - len(self.cod):]
        for dom_wires, cod_wires in self.box_wires:
            inputs += cod_wires
            outputs += dom_wires
        return sorted(inputs) == sorted(outputs)\
            == list(range(self.n_spiders - len(self._scalar_spiders)))

    @property
    def is_monogamous(self):
        """
        Checks monogamy, i.e. each spider is connected to two or zero ports.
        In that case, the diagram actually lives in a compact-closed category,
        i.e. it can be drawn using only swaps, cups and caps.

        Examples
        --------

        >>> x, y = types(" x y")
        >>> f = Box('f', x, y)
        >>> assert f.is_monogamous and f.transpose().is_monogamous
        >>> assert Cup(x, x).is_monogamous and Cap(x, x).is_monogamous
        >>> assert Spider(0, 0, x).is_monogamous
        >>> assert not Spider(1, 2, x).is_monogamous
        """
        return all(
            self.wires.count(i) in [0, 2] for i in range(self.n_spiders))

    @property
    def is_progressive(self):
        """
        Checks progressivity, i.e. wires are monotone w.r.t. box index.
        If the diagram is progressive, hetero-monogamous and it doesn't have
        any scalar spiders, then it actually lives in a symmetric monoidal
        category, i.e. it can be drawn using only swaps.

        Examples
        --------

        >>> x, y = types(" x y")
        >>> f = Box('f', x, y)
        >>> assert f.is_progressive
        >>> assert (f >> f[::-1]).is_progressive

        >>> cycle = Cap(x, x) >> Id(x) @ (f >> f[::-1]) >> Cup(x, x)
        >>> assert not cycle.is_progressive
        """
        scan = set(self.wires[:len(self.dom)])
        for dom_wires, cod_wires in self.box_wires:
            if not set(dom_wires) <= scan:
                return False
            scan = scan.union(set(cod_wires))
        return True

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

    def then(self, other):
        """ Composition of two hypergraph diagrams, i.e. their pushout. """
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod, boxes = self.dom, other.cod, self.boxes + other.boxes
        self_boundary = self.wires[len(self.wires) - len(self.cod):]
        other_boundary = other.wires[:len(other.dom)]
        graph = Graph([
            (("b", i), ("self", j)) for i, j in enumerate(self_boundary)
            ] + [
            (("b", i), ("other", j)) for i, j in enumerate(other_boundary)])
        components, self_pushout, other_pushout = set(), dict(), dict()
        for i, component in enumerate(connected_components(graph)):
            components.add(i)
            for case, j in component:
                if case == "self":
                    self_pushout[j] = i
                if case == "other":
                    other_pushout[j] = i
        self_spiders = set(range(self.n_spiders)) - set(self_boundary)
        self_pushout.update({
            j: len(components) + j
            for i, j in enumerate(self.wires) if j in self_spiders})
        other_spiders = set(range(other.n_spiders)) - set(other_boundary)
        other_pushout.update({
            j: len(components) + len(self_spiders) + j
            for i, j in enumerate(other.wires) if j in other_spiders})
        wires = [
            self_pushout[i] for i in self.wires[:
                len(self.wires) - len(self.cod)]]\
            + [other_pushout[i] for i in other.wires[len(other.dom):]]
        n_spiders = self.n_spiders - len(set(self_boundary))\
            + len(components) + other.n_spiders - len(set(other_boundary))
        return Diagram(dom, cod, boxes, wires, n_spiders, _scan=False)

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
        n_spiders = self.n_spiders + other.n_spiders
        return Diagram(dom, cod, boxes, wires, n_spiders, _scan=False)

    __matmul__ = tensor

    def dagger(self):
        dom, cod = self.cod, self.cod
        dom_wires = self.wires[len(self.wires) - len(self.cod):]
        box_wires = sum([
            cod_wires + dom_wires
            for dom_wires, cod_wires in self.box_wires[::-1]], [])
        cod_wires = self.wires[:len(self.dom)]
        wires = dom_wires + box_wires + cod_wires
        return Diagram(dom, cod, boxes, wires, self.n_spiders, _scan=False)

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
        data = list(map(repr, [self.dom, self.cod, self.boxes, self.wires]))
        data += [
            ", n_spiders={}".format(self.n_spiders)
            if self.scalar_spiders else ""]
        return "Diagram({}, {}, {}, {}{})".format(*data)

    transpose = rigid.Diagram.transpose


class Box(cat.Box, Diagram):
    """ Box in a :class:`discopy.hypergraph.Diagram`. """
    def __init__(self, name, dom, cod, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        boxes, spiders = [self], list(range(len(dom @ cod)))
        wires = 2 * list(range(len(dom)))\
            + 2 * list(range(len(dom), len(dom @ cod)))
        Diagram.__init__(self, dom, cod, boxes, wires)


class Id(Diagram):
    """ Identity diagram. """
    def __init__(self, dom):
        super().__init__(dom, dom, [], 2 * list(range(len(dom))))


class Swap(Diagram):
    """ Swap diagram. """
    def __init__(self, left, right):
        dom, cod = left @ right, right @ left
        boxes, wires = [], list(range(len(dom)))\
            + list(range(len(left), len(dom))) + list(range(len(left)))
        super().__init__(dom, cod, boxes, wires)


class Spider(Diagram):
    """ Spider diagram. """
    def __init__(self, n_legs_in, n_legs_out, typ):
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        boxes, n_spiders = [], len(typ)
        wires = (n_legs_in + n_legs_out) * list(range(n_spiders))
        super().__init__(dom, cod, boxes, wires, n_spiders)


class Cup(Diagram):
    """ Cup diagram. """
    def __init__(self, left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        super().__init__(left @ right, Ty(), [], wires)


class Cap(Diagram):
    """ Cap diagram. """
    def __init__(self, left, right):
        if not left.r == right:
            raise AxiomError
        wires = list(range(len(left))) + list(reversed(range(len(left))))
        super().__init__(Ty(), left @ right, [], wires)



Diagram.id = Id
Diagram.swap = Swap
Diagram.spiders = Spider
Diagram.cups, Diagram.caps = Cup, Cap
