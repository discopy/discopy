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
    TermBase
    Constant
    Variable
    Application
    Abstraction

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

from abc import abstractproperty, abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing import Callable, Optional, ClassVar

from discopy import cat, monoidal
from discopy.abc import BiclosedCategory
from discopy.drawing import Drawing
from discopy.cat import ob_factory, ar_factory
from discopy.utils import (
    assert_isinstance,
    factory_name,
    from_tree,
)


@ob_factory
class Ty(monoidal.Ty):
    """
    A biclosed type is a monoidal type that can be exponentiated.

    Parameters:
        inside (Ty) : The objects inside the type.

    Note
    ----
    Applying a biclosed type to a callable yields a :class:`Abstraction`.
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else monoidal.Ty.__pow__(self, other)

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(other, self)

    def __call__(self, arg):
        if isinstance(arg, Callable):
            parameters = dict(signature(arg).parameters)
            left = False
            if "left" in parameters:
                left_param = parameters.pop("left")
                left = left_param.default
                if not isinstance(left, bool):
                    raise NotImplementedError
            varnames = list(parameters.keys())
            if len(varnames) != 1:
                raise NotImplementedError
            var = self.variable_factory(varnames[0], self)
            return self.abstraction_factory(var, arg(var), left)
        raise ValueError

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


@ar_factory
class Diagram(monoidal.Diagram, BiclosedCategory):
    """
    A biclosed diagram is a monoidal diagram
    with :class:`Curry` and :class:`Eval` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """

    ob = Ty

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

    @property
    def drawing_name(self):
        return "<<" if self.left else ">>"


class Coeval(Box):
    """
    The coevaluation of an exponential type, i.e. the dagger of :class:`Eval`.

    Parameters:
        x : The exponential type to coevaluate.
    """
    drawing_name = "lambda"

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


Id = Diagram.id
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.over, Diagram.under, Diagram.exp\
    = map(staticmethod, (Over, Under, Exp))
Diagram.sum_factory = Sum


class Functor(monoidal.Functor):
    """
    A biclosed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, TermBase):
            return other.eval(self)
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


class TermBase:
    """
    A term in the internal language of biclosed categories.

    Attributes:
        typ (Ty): The type of a term, i.e. the codomain of its morphism.

    Note
    ----
    Constant terms can be instantiated from any diagram, if the domain is not
    empty (i.e. the diagram is a process not a state) then the constant is a
    given a function type with the argument coming either the left or right:

    >>> X, Y = Ty("X"), Ty("Y")
    >>> x = Constant(Box("x", Ty(), X))
    >>> f, g = Constant(Box("f", X, Y)), Constant(Box("g", X, Y), left=True)
    >>> assert x.typ == X and f.typ == X >> Y and g.typ == Y << X

    Terms can be the :class:`Application` of a function to an argument from its
    left ``>>`` or right ``<<`` with the type inferred automatically e.g.

    >>> xf, gx = x >> f, g << x
    >>> assert xf.typ == Y == gx.typ

    Applying a biclosed type to a function yields an :class:`Abstraction` e.g.

    >>> f_, g_ = X(lambda x, left=True: x >> f), X(lambda x: g << x)
    >>> assert f.typ == f_.typ == X >> Y and g.typ == g_.typ == Y << X

    Terms are required to be linear and planar, they can be drawn as diagrams:

    >>> I, N, S = Ty(), Ty("N"), Ty("S")
    >>> Alice, Bob = Constant(Box("Alice", I, N)), Constant(Box("Bob", I, N))
    >>> loves = Constant(Box("loves", N, N >> S), left=True)
    >>> (Alice >> (loves << Bob)).draw(
    ...     path='docs/_static/biclosed/alice-loves-bob.png',
    ...     margins=(.3, 0), figsize=(5, 4))
    """
    typ: Ty
    functor: ClassVar[Functor] = Functor.id(Diagram)

    @abstractproperty
    def freevars(self) -> list[Variable]:
        "The list of all occurrences of free variables, i.e. with duplicates."

    @abstractmethod
    def eval(functor: Functor = None) -> BiclosedCategory:
        """
        The evaluation of a :class:`Functor` on a term gives a morphism in its
        codomain. By default, this is the identity functor on the free biclosed
        category, i.e. terms are compiled to diagrams with constants as boxes.
        """

    def draw(self, **kwargs):
        "Drawing a term by evaluating it in the free biclosed category."
        return self.eval().draw(**kwargs)

    def __lshift__(self, other):
        return Application(self, other, left=True)

    def __rshift__(self, other):
        return Application(other, self, left=False)


@dataclass(frozen=True)
class Constant(TermBase):
    """
    A constant term of defined by a :class:`Diagram` with ``dom=X, cod=Y``.
    The constant has type ``Y`` if ``X`` is empty else it has type either
    ``Y << X`` if ``left=True`` else ``X >> Y``.

    Attributes:
        inside (Diagram): The diagram which defines the constant.
        left (Optional[bool]): Whether the domain comes from the left or right.
    """
    inside: Diagram
    left: Optional[bool] = None

    def __post_init__(self):
        if self.left is None and self.inside.dom:
            object.__setattr__(self, "left", False)

    def __str__(self):
        return f"Constant({self.inside}{', left=True' if self.left else ''})"

    @property
    def typ(self) -> Ty:
        if self.left is None:
            return self.inside.cod
        dom, cod = self.inside.dom, self.inside.cod
        return cod << dom if self.left else dom >> cod

    @property
    def freevars(self):
        return []

    def eval(self, functor=None):
        functor = functor or self.functor
        arg = self.inside if self.left is None else self.inside.curry(
            n=len(self.inside.dom), left=self.left)
        return functor(arg)


@dataclass(frozen=True)
class Variable(TermBase):
    """
    A variable with a string as name and a :class:`Ty`.

    Attributes:
        name (str): The name of the variable
        typ (Ty): The type of the variable.
    """
    name: str
    typ: Ty

    def __str__(self):
        return self.name

    @property
    def freevars(self):
        return [self]

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.cod.id(functor(self.typ))


@dataclass(frozen=True)
class Application(TermBase):
    """
    The application ``x >> f`` (``f << x``) of a term ``f`` of type ``X >> Y``
    (``Y << X``) to an argument of type ``X`` coming from its left (or right).

    Attributes:
        func (Term): The function being applied.
        args (Term): The arguments to which the function is applied.
        left (bool): Whether the argument comes in from the left or right.
    """
    func: Term
    args: Term
    left: bool = True

    def __post_init__(self):
        exp = Over if self.left else Under
        assert_isinstance(self.func.typ, exp)
        if self.func.typ.exponent != self.args.typ:
            raise ValueError(
                f"Expected {self.func.typ.exponent}, got {self.args.typ}")
        if set(self.func.freevars).intersection(self.args.freevars):
            raise ValueError("Expected disjoint free variables.")

    @property
    def typ(self):
        return self.func.typ.base

    def __str__(self):
        func, args = self.func, self.args
        func = f"({func})" if isinstance(func, Application) else str(func)
        args = f"({args})" if isinstance(args, Application) else str(args)
        return f"{func} << {args}" if self.left else f"{args} >> {func}"

    @property
    def freevars(self):
        return self.func.freevars + self.args.freevars if self.left\
            else self.args.freevars + self.func.freevars

    def eval(self, functor=None):
        functor = functor or self.functor
        func = self.func.eval(functor=functor)
        args = self.args.eval(functor=functor)
        base, exponent = self.func.typ.base, self.func.typ.exponent
        ev = functor.cod.ev(functor(base), functor(exponent), left=self.left)
        return func @ args >> ev if self.left else args @ func >> ev


@dataclass(frozen=True)
class Abstraction(TermBase):
    var: Variable
    body: Term
    left: bool = False

    def __post_init__(self):
        if self.body.freevars.count(self.var) != 1:
            raise ValueError("Expected variable to occur exactly once.")
        index = self.body.freevars.index(self.var)
        if self.left and index != 0:
            raise ValueError("Expected abstraction of left-most variable.")
        if not self.left and index != len(self.body.freevars) - 1:
            raise ValueError("Expected abstraction of right-most variable.")

    @property
    def typ(self):
        return self.var.typ >> self.body.typ if self.left\
            else self.body.typ << self.var.typ

    def __str__(self):
        left = ", left=True" if self.left else ""
        return f"{self.var.typ}(lambda {self.var.name}{left}: {self.body})"

    @property
    def freevars(self):
        return list(filter(lambda x: x != self.var, self.body.freevars))

    def to_diagram(self, **kwargs):
        return self.body.to_diagram(**kwargs).curry(left=not self.left)


type Term = Constant | Variable | Application | Abstraction

Ty.variable_factory = Variable
Ty.abstraction_factory = Abstraction
