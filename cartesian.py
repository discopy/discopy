# -*- coding: utf-8 -*-
"""
Implements the free Cartesian category and functors into Python.

>>> assert Diagram.swap(1, 2) == SWAP @ Id(1) >> Id(1) @ SWAP
>>> assert Diagram.delete(2) == DEL @ DEL
>>> assert Diagram.copy(2) == COPY @ COPY >> Id(1) @ SWAP @ Id(1)

The call method for diagrams of functions is implemented using PythonFunctors.

We can check naturality of the Swap on specific inputs:

>>> f = disco(2, 2)(lambda x, y: (x + 1, y - 1))
>>> g = disco(2, 2)(lambda x, y: (2 * x, 3 * y))
>>> assert (f @ g >> Diagram.swap(2, 2))(42, 43, 44, 45)\\
...     == (Diagram.swap(2, 2) >> g @ f)(42, 43, 44, 45)

As well as the Yang-Baxter equation:

>>> assert (SWAP @ Id(1) >> Id(1) @ SWAP >> SWAP @ Id(1))(41, 42, 43)\\
...     == (Id(1) @ SWAP >> SWAP @ Id(1) >> Id(1) @ SWAP)(41, 42, 43)

We can check the axioms for the Copy/Diagram.discard comonoid on specific inputs:

>>> assert (f >> Copy(2))(42, 43) == (Copy(2) >> f @ f)(42, 43)
>>> assert (Copy(3) >> Id(3) @ Diagram.discard(3))(42, 43, 44) == Id(3)(42, 43, 44)\\
...     == (Copy(3) >> Diagram.discard(3) @ Id(3))(42, 43, 44)
>>> assert (Copy(4) >> Swap(4, 4))(42, 43, 44, 45) == Copy(4)(42, 43, 44, 45)
"""

from discopy.cat import AxiomError
from discopy import messages, moncat
from discopy.cat import Ob, Quiver
from discopy.moncat import Ty, PRO


class Diagram(moncat.AbstractDiagram):
    """
    Implements diagrams of Python functions.
    """
    def __init__(self, dom, cod, boxes, offsets, layers=None):
        super().__init__(PRO(dom), PRO(cod), boxes, offsets, layers=layers)

    @staticmethod
    def _upgrade(diagram):
        """
        Takes a moncat.Diagram and returns a cartesian.Diagram.
        """
        return Diagram(len(diagram.dom), len(diagram.cod),
                       diagram.boxes, diagram.offsets, layers=diagram.layers)

    @staticmethod
    def id(x):
        """
        >>> Diagram.id(2)
        Id(2)
        """
        return Id(x)

    @staticmethod
    def swap(left, right):
        """
        Implements the swap function from left @ right to right @ left

        >>> assert Diagram.swap(2, 3)(0, 1, 2, 3, 4) == (2, 3, 4, 0, 1)
        """
        dom, cod = PRO(left) @ PRO(right), PRO(right) @ PRO(left)
        boxes = [SWAP for i in range(left) for j in range(right)]
        offsets = [left + i - 1 - j for j in range(left) for i in range(right)]
        return Diagram(dom, cod, boxes, offsets)

    @staticmethod
    def copy(dom):
        """
        Implements the copy function from dom to 2*dom.

        >>> assert Diagram.copy(3)(0, 1, 2) == (0, 1, 2, 0, 1, 2)
        """
        result = Id(0)
        for i in range(dom):
            result = result @ COPY
        for i in range(1, dom):
            swaps = Id(0)
            for j in range(dom - i):
                swaps = swaps @ SWAP
            result = result >> Id(i) @ swaps @ Id(i)
        return result

    @staticmethod
    def delete(dom):
        """
        Implements the discarding function on dom inputs.

        >>> assert Diagram.delete(3)(0, 1, 2) == () == Diagram.delete(2)(43, 44)
        """
        result = Id(0)
        for i in range(dom):
            result = result @ DEL
        return result

    def __call__(self, *values):
        """
        Call method implemented using PythonFunctors.

        >>> assert SWAP(1, 2) == (2, 1)
        >>> assert (COPY @ COPY >> Id(1) @ SWAP @ Id(1))(1, 2) == (1, 2, 1, 2)
        """
        return PythonFunctor(self)(*values)


class Id(Diagram):
    """
    Implements identity diagrams on dom inputs.

    >>> c =  SWAP >> ADD >> COPY
    >>> assert Id(2) >> c == c == c >> Id(2)
    """
    def __init__(self, dom):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        super().__init__(PRO(dom), PRO(dom), [], [], layers=None)

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(len(self.dom))

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)


dclass Box(moncat.Box, Diagram):
    """
    Implements generators of the free Cartesian category.
    """
    def __init__(self, name, dom, cod, data=None):
        """
        >>> assert COPY.dom == PRO(1)
        >>> assert COPY.cod == PRO(2)
        """
        moncat.Box.__init__(self, name, PRO(dom), PRO(cod), data=data)
        Diagram.__init__(self, dom, cod, [self], [0])


class Swap(Box):
    def __init__(self):
        super().__init__('SWAP', 2, 2)


class Copy(Box):
    def __init__(self):
        super().__init__('COPY', 1, 2)


class Del(Box):
    def __init__(self):
        super().__init__('DEL', 1, 0)


SWAP, COPY, DEL, ADD = Swap(), Copy(), Del(), Box('ADD', 2, 1)


class Functor(moncat.Functor):
    """
    Implements Cartesian functors.
    """
    def __init__(self, ob, ar, ob_factory=PRO, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, Swap):
            left, right = map(PRO, diagram.dom)
            return self.ar_factory.swap(self(left), self(right))
        if isinstance(diagram, Copy):
            return self.ar_factory.copy(self(diagram.dom))
        if isinstance(diagram, Del):
            return self.ar_factory.delete(self(diagram.dom))
        if isinstance(diagram, Diagram):
            return Diagram._upgrade(super().__call__(diagram))
        return super().__call__(diagram)


def tuplify(xs):
    return xs if isinstance(xs, tuple) else (xs, )


def untuplify(*xs):
    return xs[0] if len(xs) == 1 else xs


class Function(Box):
    """
    Wraps python functions with domain and codomain information.

    Parameters
    ----------
    dom : int
        Domain of the function, i.e. number of input arguments.
    cod : int
        Codomain of the diagram.
    function: any
        Python function with a call method.

    Example
    -------

    >>> sort = Function(3, 3, lambda *xs: tuple(sorted(xs)))
    >>> assert (sort >> Id(1) @ SWAP)(3, 2, 1) == (1, 3, 2)
    """
    def __init__(self, function, dom, cod):
        self._function = function
        super().__init__(repr(function), PRO(dom), PRO(cod))

    def __repr__(self):
        return "Function({}, dom={}, cod={})".format(
            repr(self._function), self.dom, self.cod)

    def __str__(self):
        return repr(self)

    def __call__(self, *values):
        return self._function(*values)

    def then(self, other):
        """
        Implements the sequential composition of Python functions.

        >>> assert (COPY >> SWAP)(1) == COPY(1)
        >>> assert (SWAP >> SWAP)(1, 2) == (1, 2)
        """
        if not isinstance(other, Diagram):
            raise TypeError(messages.type_err(Diagram, other))
        if len(self.cod) != len(other.dom):
            raise AxiomError(messages.does_not_compose(self, other))
        return Function(lambda *vals: other(*tuplify(self(*vals))),
                        self.dom, other.cod)

    def tensor(self, other):
        """
        Implements the product of Python functions.

        >>> assert (SWAP @ SWAP)(1, 2, 3, 4) == (2, 1, 4, 3)
        >>> assert (COPY @ COPY)(1, 2) == (1, 1, 2, 2)
        """
        if not isinstance(other, Diagram):
            raise TypeError(messages.type_err(Diagram, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod

        def product(*vals):
            vals0 = tuplify(self(*vals[:len(self.dom)]))
            vals1 = tuplify(other(*vals[len(self.dom):]))
            return untuplify(*(vals0 + vals1))
        return Function(product, dom, cod)

    @staticmethod
    def id(dom):
        """
        Implements the identity function on 'dom' inputs.

        >>> assert Id(0)() == ()
        >>> assert Id(2)(1, 2) == (1, 2)
        """
        return Function(untuplify, dom, dom)

    @staticmethod
    def swap(left, right):
        return right, left

    @staticmethod
    def copy(dom):
        return dom, dom

    @staticmethod
    def delete(dom):
        return ()


PythonFunctor = Functor(lambda x: x, lambda box: box, ar_factory=Function)


def disco(dom, cod):
    """
    Decorator turning a python function into a cartesian.Box storing it,
    given domain and codomain information.

    >>> @disco(2, 1)
    ... def add(x, y):
    ...     return x + y
    >>> assert isinstance(add, Box)
    """
    def decorator(func):
        return Function(func, dom, cod)
    return decorator
