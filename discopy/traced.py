from discopy import symmetric
from discopy.cat import Category
from discopy.symmetric import Ty


class Diagram(symmetric.Diagram):
    def trace(self, n=1):
        return Trace(self, n)

class Box(symmetric.Box, Diagram):
    cast = Diagram.cast

class Trace(Box):
    def __init__(self, diagram: Diagram, n=1):
        self.diagram, name = diagram, "Trace({}, {})".format(diagram, n)
        super().__init__(name, diagram.dom[:-n], diagram.cod[:-n])

class Functor(symmetric.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.diagram.dom)) - len(self(other.dom))
            return self.cod.ar.trace(self(other.diagram), n)
        return super().__call__(other)
