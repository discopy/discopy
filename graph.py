from moncat import Type, Diagram, Box, MonoidalFunctor
import pyzx as zx
import networkx as nx

B, Z, X = 0, 1, 2

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

    def __eq__(self,other):
        if not isinstance(other, OpenGraph):
            return False
        if not self.dom == other.dom and self.cod == other.cod:
            return False
        if not set(self.graph.nodes()) == set(other.graph.nodes()):
            return False
        return set(self.graph.edges()) == set(other.graph.edges())

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

    def to_zx(self):
        k = zx.Graph()
        for x in self.graph.nodes(data = True):
            k.add_vertex(ty = x[1]['ty'])
        k.add_edges(list(self.graph.edges()))
        return k

    @staticmethod
    def from_zx(dom, cod, d):
        g = nx.MultiGraph()
        for x, y in d.edges():
            g.add_edge(x,y)
        for i in d.vertices():
            g.add_node(i, ty = d.types()[i])
        return OpenGraph(dom , cod, g)

class Node(OpenGraph):
    def __init__(self, dom, cod, label):
        g = nx.MultiGraph()
        g.add_nodes_from(range(dom + 1 +cod), ty = B) #boundary nodes
        g.add_node(dom, ty = label) #the inner node is labeled
        g.add_edges_from([(i, dom) for i in range(dom)])
        g.add_edges_from([(dom, dom + 1 + j) for j in range(cod)])
        super().__init__(dom, cod, g)

class IdGraph(OpenGraph):
    def __init__(self, dom):
        g = nx.MultiGraph()
        g.add_nodes_from(range(dom + dom), ty = B)
        g.add_edges_from([(i, dom + i) for i in range(dom)])
        super().__init__(dom, dom, g)

class GraphFunctor(MonoidalFunctor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Type) and len(x) == 1 for x in ob.keys())
        assert all(isinstance(a, Box) for a in ar.keys())
        assert all(isinstance(b, OpenGraph) for b in ar.values())
        self._ob, self._ar = {x[0]: y for x, y in ob.items()}, ar

    def __call__(self, d):
        if isinstance(d,Type):
            return sum([self.ob[x] for x in d])

        if isinstance(d, Box):
            return self.ar[d]

        if isinstance(d, Diagram):
            u = d.dom
            g = IdGraph(self(u))
            for f, n in zip(d.boxes, d.offsets):
                g = g.then(IdGraph(self(u[:n])).tensor(self(f))\
                     .tensor(IdGraph(self(u[n + len(f.dom):]))))
                u = u[:n] + f.cod + u[n + len(f.dom):]
            return g

x, y, z, w = Type('x'), Type('y'), Type('z'), Type('w')
f, g, h = Box('f', x, x + y), Box('g', y + z, w), Box('h', x + w, x)
diagram = f.tensor(Diagram.id(z)).then(Diagram.id(x).tensor(g))

ob = {x: 1, y: 2, z: 3, w: 4}
D = {f: Node(sum(ob[Type([x])] for x in f.dom), sum(ob[Type([b])] for b in f.cod), Z),
     g: Node(sum(ob[Type([x])] for x in g.dom), sum(ob[Type([b])] for b in g.cod), X) }

F = GraphFunctor(ob, D)

opengraph = D[f].tensor(IdGraph(F(z))).then(IdGraph(F(x)).tensor(D[g]))
assert opengraph == F(diagram)

C = zx.generate.cnots(3,4)
assert OpenGraph.from_zx( 3,3, OpenGraph.from_zx(3, 3, C).to_zx()) == OpenGraph.from_zx(3,3,C)
