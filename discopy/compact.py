from discopy import symmetric, tortile
from discopy.cat import Category
from discopy.tortile import Ty, hexagon, nesting


class Diagram(symmetric.Diagram, tortile.Diagram):
    def trace(self, n=1):
        return self.dom[:-n] @ self.caps(self.dom[-n:], self.dom[-n:].r)\
            >> self @ self.dom[-n:].r\
            >> self.cod[:-n] @ self.cups(self.cod[-n:], self.cod[-n:].r)

class Box(symmetric.Box, tortile.Box, Diagram):
    cast = Diagram.cast

class Cup(tortile.Cup, Box): pass

class Cap(tortile.Cap, Box): pass

class Swap(symmetric.Swap, tortile.Braid, Box): pass

Diagram.braid = Diagram.swap = hexagon(Swap)
Diagram.cups, Diagram.caps = nesting(Cup), nesting(Cap)

class Functor(symmetric.Functor, tortile.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return symmetric.Functor.__call__(self, other)
        return tortile.Functor.__call__(self, other)
