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
from networkx import DiGraph as Graph, subgraph_view as subgraph

from discopy import cat, monoidal
from discopy.cat import AxiomError
from discopy.monoidal import Ty, types
from discopy.drawing import Node


Graph.relabel = nx.relabel_nodes


class Diagram(cat.Arrow):
    """
    Diagram in a symmetric monoidal category.

    >>> x, y, z = types("x y z")
    >>> f, g = Box("f", x, y @ z), Box("g", z @ y, x)
    >>> diagram = f >> Swap(y, z) >> g
    >>> assert set(diagram.nodes) == set(range(
    ...     len(diagram.dom)
    ...     + sum(len(box.dom @ box.cod) for box in diagram.boxes)
    ...     + len(diagram.cod)))
    >>> diagram.edges
    OutEdgeView([(0, 1), (2, 5), (3, 4), (6, 7)])
    """
    def __init__(self, dom, cod, boxes, graph, _scan=True):
        super().__init__(dom, cod, boxes, _scan=False)
        if _scan:
            assert sorted(list(graph.nodes)) == list(range(
                len(dom) + sum(len(box.dom) for box in boxes)
                + sum(len(box.cod) for box in boxes) + len(cod)))

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
        return "Diagram({}, {}, {}, nx.DiGraph({}))".format(*map(repr, [
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
            source, = graph.predecessors(i)
            target, = graph.successors(i)
            graph.add_edge(source, target)
            graph.remove_node(i)
        graph.relabel(copy=False, mapping={
            i: i - len(boundary) for i in graph.nodes if i > boundary[-1]})
        return Diagram(dom, cod, boxes, graph)

    def tensor(self, other):
        """ Tensor product of two symmetric monoidal diagrams. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        # move self.cod to the end
        self_graph = self.graph.relabel({
            i: len(other.nodes) - len(other.cod) + i
            for i in range(len(self.nodes) - len(self.cod), len(self.nodes))})
        # make space for other.dom
        self_graph.relabel({i: i + len(other.dom) for i in range(
            len(self.dom), len(self.nodes) - len(self.cod))}, copy=False)
        # move other.dom to the start, other.cod to the end
        # and the nodes for other.boxes to just before self.cod
        other_graph = other.graph.relabel({
            i: len(self.dom) + i if i < len(other.dom)
            else len(self.nodes) + i if i >= len(other.nodes) - len(other.cod)
            else len(self.nodes) - len(self.cod) + i
            for i in range(len(other.nodes))})
        graph = nx.union(self_graph, other_graph)
        return Diagram(dom, cod, boxes, graph)

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
        boxes, graph = [self], Graph(
            [(i, len(dom) + i) for i, _ in enumerate(dom)] + [
                (len(dom @ dom) + i, len(dom @ dom @ cod) + i)
                for i, _ in enumerate(cod)])
        Diagram.__init__(self, dom, cod, [self], graph)


class Swap(Diagram):
    """ Swap in a symmetric monoidal diagram. """
    def __init__(self, left, right):
        dom, cod = left @ right, right @ left
        boxes, graph = [], Graph(
            [(i, len(dom @ right) + i) for i, _ in enumerate(left)] + [
                (len(left) + i, len(dom) + i) for i, _ in enumerate(right)])
        super().__init__(dom, cod, boxes, graph)


class Id(Diagram):
    """ Identity diagram. """
    def __init__(self, dom):
        dom, cod, boxes, graph = dom, dom, [], Graph(
            [(i, len(dom) + i) for i, _ in enumerate(dom)])
        super().__init__(dom, cod, boxes, graph)
