# -*- coding: utf-8 -*-

"""
The free biclosed monoidal category, i.e. with left and right exponentials.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Over
    Under
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
    Functor

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, z << y), Box('g', x @ y, z), Box('h', y, x >> z)

>>> from discopy.drawing import Equation
>>> Equation(f.uncurry(left=True).curry(left=True), f).draw(
...     path='docs/_static/biclosed/curry-left.png', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-left.png
    :align: center

>>> Equation(h.uncurry().curry(), h).draw(
...     path='docs/_static/biclosed/curry-right.png', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-right.png
    :align: center

>>> Equation(
...     g.curry(left=True).uncurry(left=True), g, g.curry().uncurry()).draw(
...         path='docs/_static/biclosed/uncurry.png')

.. image:: /_static/biclosed/uncurry.png
    :align: center
"""

from __future__ import annotations

from discopy import cat, monoidal
from discopy.abc import BiclosedCategory
from discopy.drawing import Drawing
from discopy.cat import factory
from discopy.utils import (
    factory_name,
    from_tree,
)


@factory
class Ty(monoidal.Ty):
    """
    A biclosed type is a monoidal type that can be exponentiated.

    Parameters:
        inside (Ty) : The objects inside the type.

    Note
    ----
    We can exponentials of types.

    >>> x, y, z = Ty(*"xyz")
    >>> print((x ** y) ** z)
    ((x ** y) ** z)

    We can also distinguish left- and right-exponentials.

    >>> print((x >> y) << z)
    ((x >> y) << z)
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(other, self)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.inside))})"

    @property
    def left(self) -> Ty:
        return self.inside[0].left if self.is_exp else None

    @property
    def right(self) -> Ty:
        return self.inside[0].right if self.is_exp else None

    @property
    def is_exp(self):
        """
        Whether the type is an :class:`Exp` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x ** y).is_exp and (x ** y @ Ty()).is_exp
        """
        return len(self) == 1 and isinstance(self.inside[0], Exp)

    @property
    def is_under(self):
        """
        Whether the type is an :class:`Under` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x >> y).is_under and (x >> y @ Ty()).is_under
        """
        return len(self) == 1 and isinstance(self.inside[0], Under)

    @property
    def is_over(self):
        """
        Whether the type is an :class:`Over` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x << y).is_over and (x << y @ Ty()).is_over
        """
        return len(self) == 1 and isinstance(self.inside[0], Over)


class Exp(Ty, cat.Ob):
    """
    A :code:`base` type to an :code:`exponent` type, called with :code:`**`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """

    def __init__(self, base: Ty, exponent: Ty):
        self.base, self.exponent = base, exponent
        super().__init__(self)

    @property
    def left(self):
        return self.exponent if isinstance(self, Under) else self.base

    @property
    def right(self):
        return self.base if isinstance(self, Under) else self.exponent

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.base, self.exponent) == (other.base, other.exponent)
        if isinstance(other, Exp):
            return False  # Avoid infinite loop with Over(x, y) == Under(x, y).
        return isinstance(other, Ty) and other.inside == (self, )

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return f"({self.base} ** {self.exponent})"

    def __repr__(self):
        return factory_name(type(self))\
            + f"({repr(self.base)}, {repr(self.exponent)})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'base': self.base.to_tree(),
            'exponent': self.exponent.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['base'], tree['exponent'])))

    def to_drawing(self):
        return Ty(str(self)).to_drawing()


class Over(Exp):
    """
    An :code:`exponent` type over a :code:`base` type, called with :code:`<<`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return f"({self.base} << {self.exponent})"


class Under(Exp):
    """
    A :code:`base` type under an :code:`exponent` type, called with :code:`>>`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return f"({self.exponent} >> {self.base})"


@factory
class Diagram(monoidal.Diagram, BiclosedCategory):
    """
    A biclosed diagram is a monoidal diagram
    with :class:`Curry` and :class:`Eval` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """

    ty_factory = Ty

    def curry(self, n=1, left=False) -> Diagram:
        """
        Wrapper around :class:`Curry` called by :class:`Functor`.

        Parameters:
            n : The number of atomic types to curry.
            left : Whether to curry on the left or right.
        """
        return self.curry_factory(self, n, left)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=False) -> Eval:
        """
        Wrapper around :class:`Eval` called by :class:`Functor`.

        Parameters:
            base : The base of the exponential type to evaluate.
            exponent : The exponent of the exponential type to evaluate.
            left : Whether to evaluate on the left or right.
        """
        return cls.eval_factory(
            base << exponent if left else exponent >> base)

    def uncurry(self: Diagram, left=False) -> Diagram:
        """
        Uncurry a biclosed diagram by composing it with :meth:`Diagram.ev`.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        base, exponent = self.cod.base, self.cod.exponent
        return self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)


class Box(monoidal.Box, Diagram):
    """
    A biclosed box is a monoidal box in a biclosed diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """


class Eval(Box):
    """
    The evaluation of an exponential type.

    Parameters:
        x : The exponential type to evaluate.
    """
    def __init__(self, x: Exp, left=None):
        self.x, self.left = x, isinstance(x, Over) if left is None else left
        dom, cod = (x @ x.exponent, x.base) if self.left\
            else (x.exponent @ x, x.base)
        super().__init__("Eval" + str(x), dom, cod)

    def dagger(self) -> Coeval:
        return self.coeval_factory(self.x, self.left)


class Coeval(Box):
    """
    The coevaluation of an exponential type, i.e. the dagger of :class:`Eval`.

    Parameters:
        x : The exponential type to coevaluate.
    """
    def __init__(self, x: Exp, left=None):
        self.x, self.left = x, isinstance(x, Over) if left is None else left
        cod, dom = (x @ x.exponent, x.base) if self.left\
            else (x.exponent @ x, x.base)
        super().__init__("Coeval" + str(x), dom, cod)

    def dagger(self) -> Eval:
        return self.eval_factory(self.x, self.left)


class Curry(monoidal.Bubble, Box):
    """
    The currying of a biclosed diagram.

    Parameters:
        arg : The diagram to curry.
        n : The number of atomic types to curry.
        left : Whether to curry on the left or right.
    """
    def __init__(self, arg: Diagram, n=1, left=False):
        self.n, self.left = n, left
        name = f"Curry({arg}, {n}, {left})"
        if left:
            dom = arg.dom[:len(arg.dom) - n]
            cod = arg.cod << arg.dom[len(arg.dom) - n:]
        else:
            dom, cod = arg.dom[n:], arg.dom[:n] >> arg.cod
        monoidal.Bubble.__init__(
            self, arg, dom=dom, cod=cod, drawing_name="$\\Lambda$")
        Box.__init__(self, name, dom, cod)


class Sum(monoidal.Sum, Box):
    """
    A biclosed sum is a monoidal sum and a biclosed box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


Diagram.over, Diagram.under, Diagram.exp\
    = map(staticmethod, (Over, Under, Exp))
Diagram.sum_factory = Sum

Id = Diagram.id


class Functor(monoidal.Functor):
    """
    A biclosed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ty_factory`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls) and hasattr(self.cod, attr):
                method = getattr(self.cod, attr)
                return method(self(other.base), self(other.exponent))
        if isinstance(other, Curry) and hasattr(self.cod, "curry"):
            return self.cod.curry(
                self(other.arg), len(self(other.cod.exponent)), other.left)
        if isinstance(other, (Eval, Coeval)) and hasattr(self.cod, "ev"):
            base, exponent, left = other.x.base, other.x.exponent, other.left
            result = self.cod.ev(self(base), self(exponent), left)
            return result.dagger() if isinstance(other, Coeval) else result
        if self.cod is Drawing:
            if isinstance(other, Ty) and other.inside == (other, ):
                # Avoid infinite recursion when drawing.
                return self.ob_map[other]
        return super().__call__(other)


def to_rigid(self):
    from discopy import rigid

    return Functor(
        ob=lambda x: rigid.Ty(x.inside[0].name),
        ar=lambda f: rigid.Box(
            f.name, Diagram.to_rigid(f.dom), Diagram.to_rigid(f.cod)),
        cod=rigid.Diagram)(self)


Id = Diagram.id
Diagram.to_rigid = to_rigid
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
