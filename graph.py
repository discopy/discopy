from category import Arrow, Generator, Identity, Functor


class OpenGraph(Arrow):
    def __init__(self, dom, cod, nodes, edges):
        self.dom, self.cod, self.nodes, self.edges = dom, cod, nodes, edges

    def __eq__(self, other):
        assert isinstance(other, OpenGraph)
        return self.nodes == other.nodes and self.edges == other.edges\
            and super().__eq__(other)

    def __repr__(self):
        return "OpenGraph"

    def then(self, other):
        return OpenGraph()

    def tensor(self, other):
        return OpenGraph()


class Node(Generator, OpenGraph):
    pass

class Edge(Identity, OpenGraph):
    pass
