from category import Arrow, Generator, Identity, Functor
from diagram import Diagram
import pyzx as zx
import networkx as nx

class OpenGraph(nx.Graph):
    def __init__(self, dom, cod, graph):
        assert isinstance(graph, nx.Graph)
        assert all(x not in cod for x in dom) and all(x not in dom for x in cod)
        assert all(x in graph.nodes() for x in dom + cod)
        assert all(graph.degree(x) == 1 for x in dom + cod)
        self.dom, self.cod, self.graph = dom, cod, graph
        super().__init__(graph)

    def then(self, other):
        assert isinstance(other, OpenGraph) and self.cod == other.dom
        g = self.graph.copy()

class Node(OpenGraph):
    def __init__(self, name, pos, data=None):
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        super().__init__([0], [2], graph)

    def __repr__(self):
        return 


class GraphFunctor(Functor):
    def __call__(self, d):
        if not isinstance(d,Diagram):
            return d[0]
        if isinstance(d, Diagram):
            g = nx.Graph()
            g.add_nodes_from(self(d.dom) + self(d.nodes) + self(d.cod))

            nodes = self(d.dom)
            for x, n in d.nodes, d.offsets:
                edges = [
                g = g.add_edges_from()

            return g

x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [y, z]), Box('g', [z, x], [w]), Box('h', [y, w], [x])
d = f.tensor(Wire(x)).then(Wire(y).tensor(g))

F0 = GraphFunctor({x: 1, y: 2, z: 3, w: 4}, None)
F = GraphFunctor(Fo.ob, {a: nx.Graph(F0(a.dom) + F0(a.cod)) for a in [f, g, h]})
