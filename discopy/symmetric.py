# -*- coding: utf-8 -*-

""" Symmetric monoidal categories.

Notes
-----
We can check that the axioms for symmetry hold on the nose.

>>> x, y, z = types("x y z")

Involution:

>>> assert Swap(x, y) >> Swap(y, x) == Id(x @ y)

Pentagons:

>>> assert Swap(x, y @ z) == Swap(x, y) @ Id(z) >> Id(y) @ Swap(x, z)
>>> assert Swap(x @ y, z) == Id(x) @ Swap(y, z) >> Swap(x, z) @ Id(y)

Yang-Baxter:

>>> left = Swap(x, y) @ Id(z)\\
...     >> Id(y) @ Swap(x, z)\\
...     >> Swap(y, z) @ Id(x)
>>> right = Id(x) @ Swap(y, z)\\
...     >> Swap(x, z) @ Id(y)\\
...     >> Id(z) @ Swap(x, y)
>>> assert left == right

Naturality:

>>> f = Box("f", x, y)
>>> assert f @ Id(z) >> Swap(f.cod, z) == Swap(f.dom, z) >> Id(z) @ f
"""

import networkx as nx
from networkx import Graph

from discopy import cat, monoidal
from discopy.cat import AxiomError
from discopy.monoidal import Ty, types
from discopy.drawing import Node


Graph.relabel = nx.relabel_nodes


class Diagram(cat.Arrow):
    """
    Diagram in a symmetric monoidal category.

    Parameters
    ----------
    dom : discopy.monoidal.Ty
        Domain of the diagram.
    cod : discopy.monoidal.Ty
        Codomain of the diagram.
    boxes : List[Box]
        List of :class:`discopy.symmetric.Box`.
    wires : List[Tuple[Int, Int]]
        List of edges defining the connectivity graph.

    Note
    ----

    Nodes are given by the range of::

        len(dom) + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod)

    The edges are wires (i, j) with i < j.
    The graph is monogamous, i.e. each node has degree exactly one.

    Examples
    --------

    >>> x, y, z = types("x y z")
    >>> f, g = Box("f", x, y @ z), Box("g", z @ y, x)
    >>> diagram = f >> Swap(y, z) >> g
    >>> diagram.nodes
    NodeView((0, 1, 2, 5, 3, 4, 6, 7))
    >>> diagram.edges
    EdgeView([(0, 1), (2, 5), (3, 4), (6, 7)])
    """
    def __init__(self, dom, cod, boxes, wires, _scan=True):
        super().__init__(dom, cod, boxes, _scan=False)
        graph = Graph(wires)
        if _scan:
            n_nodes = len(dom)\
                + sum(len(box.dom) + len(box.cod) for box in boxes) + len(cod)
            if set(graph.nodes) != set(range(n_nodes)):
                raise ValueError
            if set(dict(graph.degree).values()) not in (set(), {1}):
                raise ValueError
            for i, j in graph.edges:
                if i >= j:
                    raise ValueError
        self._graph = graph

    @property
    def graph(self):
        """ The graph defining a diagram. """
        return self._graph

    @property
    def nodes(self):
        """ The nodes in the graph defining a diagram. """
        return self.graph.nodes

    @property
    def edges(self):
        """ The edges in the graph defining a diagram. """
        return self.graph.edges

    def __repr__(self):
        return "Diagram({}, {}, {}, {})".format(*map(repr, [
            self.dom, self.cod, self.boxes, list(self.edges)]))

    def then(self, other):
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        graph = nx.compose(self.graph, other.graph.relabel({
            i: len(self.nodes) - len(self.cod) + i for i in other.nodes}))
        boundary = range(len(self.nodes) - len(self.cod), len(self.nodes))
        for i in boundary:
            source, target = sorted(graph.neighbors(i))
            graph.add_edge(source, target)
            graph.remove_node(i)
        graph = graph.relabel({
            i: i - len(boundary) for i in graph.nodes if i > boundary[-1]})
        return Diagram(dom, cod, boxes, graph.edges)

    def tensor(self, other):
        """ Tensor product of two symmetric monoidal diagrams. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        # move self.cod to the end
        self_graph = self.graph.relabel({
            i: len(other.nodes) - len(other.cod) + i
            for i in range(len(self.nodes) - len(self.cod), len(self.nodes))})
        # make space for other.dom
        self_graph = self_graph.relabel({i: i + len(other.dom) for i in range(
            len(self.dom), len(self.nodes) - len(self.cod))})
        # move other.dom to the start, other.cod to the end
        # and the nodes for other.boxes to just before self.cod
        other_graph = other.graph.relabel({
            i: len(self.dom) + i if i < len(other.dom)
            else len(self.nodes) + i if i >= len(other.nodes) - len(other.cod)
            else len(self.nodes) - len(self.cod) + i
            for i in range(len(other.nodes))})
        graph = nx.union(self_graph, other_graph)
        return Diagram(dom, cod, boxes, graph.edges)

    def __matmul__(self, other):
        return self.tensor(other)

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return self.dom == other.dom\
            and self.cod == other.cod\
            and self.boxes == other.boxes\
            and set(self.graph.edges) == set(other.graph.edges)


class Box(cat.Box, Diagram):
    """ Box in a symmetric monoidal diagram.

    Examples
    --------
    >>> x, y, z = types("x y z")
    >>> f = Box("f", x, y @ z)
    >>> f
    Box('f', Ty('x'), Ty('y', 'z'))
    """
    def __init__(self, name, dom, cod, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        boxes, wires = [self], [
            (i, len(dom) + i) for i, _ in enumerate(dom)] + [
            (len(dom @ dom) + i, len(dom @ dom @ cod) + i)
            for i, _ in enumerate(cod)]
        Diagram.__init__(self, dom, cod, [self], wires)


class Swap(Diagram):
    """ Swap in a symmetric monoidal diagram. """
    def __init__(self, left, right):
        dom, cod = left @ right, right @ left
        boxes, wires = [], [
            (i, len(dom @ right) + i) for i, _ in enumerate(left)] + [
            (len(left) + i, len(dom) + i) for i, _ in enumerate(right)]
        super().__init__(dom, cod, boxes, wires)


class Id(Diagram):
    """ Identity diagram. """
    def __init__(self, dom):
        super().__init__(dom, dom, [], [
            (i, len(dom) + i) for i, _ in enumerate(dom)])
