from category import Arrow, Generator, Identity, Functor
from diagram import Diagram, Box, Wire, MonoidalFunctor
import pyzx as zx
import networkx as nx

class OpenGraph:
    def __init__(self, dom, cod, graph):
        assert isinstance(graph, nx.MultiGraph)
        assert set(graph.nodes()) == set(range(len(graph.nodes())))
        assert isinstance(dom, int) and isinstance(cod, int)
        assert dom + cod <= len(graph.nodes())
        assert all(graph.degree(x) == 1 for x in range(dom))
        assert all(graph.degree(len(graph.nodes()) - cod + x) == 1 for x in range(cod))
        self.dom, self.cod, self.graph = dom, cod, graph

    def __repr__(self):
        return "OpenGraph({}, {}, {}, {})".format(
            self.dom, self.cod, self.graph.nodes(), self.graph.edges())

    def then(self, other):
        assert isinstance(other, OpenGraph) and self.cod == other.dom
        g0, g1 = self.graph.copy(), other.graph.copy()
        l0 , l1 = len(g0.nodes()), len(g1.nodes())

        g0.remove_nodes_from(range(l0 - self.cod, l0)) #remove boundary nodes
        g1.remove_nodes_from(range(other.dom))
        def relabel(x):
            return x - other.dom  + l0 - self.cod #relabeling for nodes in other.graph
        g1 = nx.relabel_nodes(g1, {x : relabel(x) for x in g1.nodes()})
        g = nx.union(g0, g1)
        for i in range(self.cod):
            source = list(self.graph.neighbors(l0 - self.cod + i))[0]
            target = relabel(list(other.graph.neighbors(i))[0])
            g.add_edge(source, target)
        return OpenGraph(self.dom, other.cod, g)

    def tensor(self, other):
        assert isinstance(other, OpenGraph)
        g0, g1 = self.graph.copy(), other.graph.copy()
        l0 , l1 = len(g0.nodes()), len(g1.nodes())
        dom0, middle0, cod0 = range(self.dom), range(self.dom, l0 - self.cod), range(l0 - self.cod, l0)
        dom1, middle1, cod1 = range(other.dom), range(other.dom, l1 - other.cod), range(l1 - other.cod, l1)

        def relabel0(x):
            if x in dom0:
                return x
            elif x in middle0:
                return x + len(dom1)
            elif x in cod0:
                return x + len(dom1) + len(middle1)

        def relabel1(x):
            if x in dom1:
                return x +  len(dom0)
            elif x in middle1:
                return x + len(dom0) + len(middle0)
            elif x in cod1:
                return x + len(dom0) + len(middle0) + len(cod0)

        g0 = nx.relabel_nodes(g0, {x : relabel0(x) for x in g0.nodes()})
        g1 = nx.relabel_nodes(g1, {x : relabel1(x) for x in g1.nodes()})

        g = nx.union(g0, g1)
        return OpenGraph(self.dom + other.dom, self.cod + other.cod, g)


class Node(OpenGraph):
    def __init__(self, dom, cod):
        g = nx.MultiGraph([(i, dom) for i in range(dom)])
        g.add_edges_from([(dom, dom + 1 + j) for j in range(cod)])
        super().__init__(dom, cod, g)

class Edge(OpenGraph):
    def __init__(self, dom):
        g = nx.MultiGraph([(i, dom + i) for i in range(dom)])
        super().__init__(dom, dom, g)

class GraphFunctor(MonoidalFunctor):
    def __call__(self, d):
        if not isinstance(d,Diagram):
            xs = d if isinstance(d, list) else [d]
            return sum([self.ob[x] for x in xs])

        if isinstance(d, Box):
            return self.ar[d]

        if isinstance(d, Diagram):
            u = d.dom
            g = Edge(self(u))

            for f, n in zip(d.nodes, d.offsets):
                g = g.then(Edge(self(u[:n])).tensor(self(f))\
                     .tensor(Edge(self(u[n + len(f.dom):]))))
                u = u[:n] + f.cod + u[n + len(f.dom):]

            return g


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [y, z]), Box('g', [z, x], [w]), Box('h', [y, w], [x])
diagram = f.tensor(Wire(x)).then(Wire(y).tensor(g))

F0 = GraphFunctor({x: 1, y: 2, z: 3, w: 4}, None)
dict = {a: Node(F0(a.dom), F0(a.cod)) for a in [f, g, h]}
F = GraphFunctor(F0.ob, dict)

print(dict[f].tensor(Edge(1)).then(Edge(2).tensor(dict[g])))
print(F(diagram))
