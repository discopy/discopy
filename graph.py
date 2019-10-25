from category import Arrow, Generator, Identity, Functor
from diagram import Diagram, Box, Wire
import pyzx as zx
import networkx as nx

class OpenGraph:
    def __init__(self, dom, cod, graph):
        assert isinstance(graph, nx.Graph)
        assert set(graph.nodes()) == set(range(len(graph.nodes())))
        assert isinstance(dom, int) and isinstance(cod, int)
        assert dom + cod <= len(graph.nodes())
        assert all(graph.degree(x) == 1 for x in range(dom))
        assert all(graph.degree(len(graph.nodes()) - cod + x) == 1 for x in range(cod))
        self.dom, self.cod, self.graph = dom, cod, graph

    def then(self, other):
        assert isinstance(other, OpenGraph) and self.cod == other.dom
        l0 , l1 = len(self.graph.nodes()), len(other.graph.nodes())
        g0, g1 = self.graph, other.graph
        g0.remove_nodes_from(range(l1 - self.cod, l1))
        g1.remove_nodes_from(range(other.dom))
        labeling1 = {x : x - other.dom  + l0 - self.cod for x in range(other.dom + 1, l1)}
        g1 = nx.relabel_nodes(g1, labeling1)
        g = nx.union(g0, g1)
        for i in range(self.cod):
            import pdb; pdb.set_trace()
            source = self.graph.neighbors(l0 - self.cod + i)[0]
            target = other.graph.neighbors(i)[0] + l0 - self.cod
            g.add_edge(source, target)
        return g

class Node(OpenGraph):
    def __init__(self, dom, cod):
        g = nx.Graph([(i, dom) for i in range(dom)])
        g.add_edges_from([(dom, dom + 1 + j) for j in range(cod)])
        super().__init__(dom, cod, g)


class GraphFunctor(Functor):
    def __call__(self, d):
        if not isinstance(d,Diagram):
            xs = d if isinstance(d, list) else [d]
            return sum([self.ob[x] for x in xs])
        if isinstance(d, Diagram):
            g = nx.Graph()
            return g

x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [y, z]), Box('g', [z, x], [w]), Box('h', [y, w], [x])
d = f.tensor(Wire(x)).then(Wire(y).tensor(g))

F0 = GraphFunctor({x: 1, y: 2, z: 3, w: 4}, None)
F = GraphFunctor(F0.ob, {a: Node(F0(a.dom), F0(a.cod)) for a in [f, g, h]})

Node(2,3).then(Node(3,1))
