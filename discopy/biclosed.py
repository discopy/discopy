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

from abc import abstractmethod
from inspect import signature
from dataclasses import dataclass
from typing import Callable, ClassVar, Self, Iterable

from discopy import cat, monoidal, messages
from discopy.abc import BiclosedCategory
from discopy.drawing import Drawing
from discopy.cat import ob_factory, ar_factory
from discopy.utils import (
    AxiomError,
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
    Applying a biclosed type to a callable yields a :class:`Abstraction`,
    applying it to a string yields a :class:`Constant`.
    """
    def __pow__(self, other: Ty) -> Ty:
        return self.exp(other) if isinstance(other, Ty)\
            else monoidal.Ty.__pow__(self, other)

    def exp(self, other: Ty) -> Ty:
        return self.ob(self.exp_factory(self, other))

    def over(self, other: Ty) -> Ty:
        return self.ob(self.over_factory(self, other))

    def under(self, other: Ty) -> Ty:
        return self.ob(self.under_factory(self, other))

    def __lshift__(self, other):
        return self.over(other)

    def __rshift__(self, other):
        return other.under(self)

    def __call__(self, arg):
        if isinstance(arg, str):
            return self.constant_factory(arg, self)
        elif isinstance(arg, Callable):
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

    def __call__(self, arg):
        if isinstance(arg, str):
            return self.constant_factory(arg, self)
        elif isinstance(arg, Callable):
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
    def is_over(self):
        """
        Whether the type is an :class:`Over` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x << y).is_over and (x << y @ Ty()).is_over
        """
        return len(self) == 1 and isinstance(self.inside[0], Over)

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
    def base(self):
        "The base of an exponential type, assumes ``self.is_exp``."
        assert self.is_exp
        return self.inside[0].base

    @property
    def exponent(self):
        "The exponent of an exponential type, assumes ``self.is_exp``."
        assert self.is_exp
        return self.inside[0].exponent


class Exp(cat.Ob):
    """
    A :code:`base` type to an :code:`exponent` type, called with :code:`**`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """

    ob = Ty

    def __init__(self, base: Ty, exponent: Ty):
        assert_isinstance(base, self.ob)
        assert_isinstance(exponent, self.ob)

        assert self.ob == base.ob == exponent.ob
        self.base, self.exponent = base, exponent
        super().__init__(str(self))

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and (self.base, self.exponent) == (other.base, other.exponent)

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return f"({self.base} ** {self.exponent})"

    def __repr__(self):
        return factory_name(type(self)) + f"({self.base!r}, {self.exponent!r})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'base': self.base.to_tree(),
            'exponent': self.exponent.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['base'], tree['exponent'])))

    @property
    def left(self):
        return self.exponent if isinstance(self, Under) else self.base

    @property
    def right(self):
        return self.base if isinstance(self, Under) else self.exponent


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
        assert x.is_exp
        self.x = x
        exp = x.inside[0]
        self.left = isinstance(exp, Over) if left is None else left
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
        assert x.is_exp
        self.x = x
        exp = x.inside[0]
        self.left = isinstance(exp, Over) if left is None else left
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

    def to_drawing(self):
        if self.left:
            f, e = self.arg, self.coeval_factory(self.cod, left=True)
            return (f >> e).to_drawing().trace()
        f, e = self.arg, self.coeval_factory(self.cod)
        return (f >> e).to_drawing().trace(left=True)


class Sum(monoidal.Sum, Box):
    """
    A biclosed sum is a monoidal sum and a biclosed box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(monoidal.Functor):
    """
    A biclosed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, TermBase):
            return other.eval(self)
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls):
                base, exponent = self(other.base), self(other.exponent)
                if hasattr(base, attr):
                    return getattr(base, attr)(exponent)
                if hasattr(self.cod, attr):
                    return getattr(self.cod, attr)(base, exponent)
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


class CMap(monoidal.CMap):
    functor = Functor


Id = Diagram.id
Diagram.map_factory = CMap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.sum_factory = Sum


class TermBase(Box):
    """
    A term in the internal language of biclosed categories.

    Attributes:
        dom (Ty): The tensor of the types for each free variable.
        cod (Ty): The type of a term, i.e. the codomain of its morphism.
        freevars (Ty): The list of free variables.
        functor (Functor): The functor to evaluate the term, ``id`` by default.

    Note
    ----
    Constant terms can be instantiated from any diagram, if the domain is not
    empty (i.e. the diagram is a process not a state) then the constant is a
    given a function type with the argument coming either the left or right:

    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, f, g = X("x"), (X >> Y)("f"), (Y << X)("g")

    Terms can be the :class:`Application` of a function to an argument from its
    left ``>>`` or right ``<<`` with the type inferred automatically e.g.

    >>> xf, gx = x(f, left=True), g(x)
    >>> assert xf.cod == Y == gx.cod

    Applying a biclosed type to a function yields an :class:`Abstraction` e.g.

    >>> f_, g_ = X(lambda y, left=True: y(f, left=True)), X(lambda y: g(y))
    >>> assert f.cod == f_.cod == X >> Y and g.cod == g_.cod == Y << X

    Terms are required to be linear and planar, they can be drawn as diagrams:

    >>> N, S = Ty("N"), Ty("S")
    >>> Alice, loves, Bob = N("Alice"), ((N >> S) << N)("loves"), N("Bob")
    >>> Alice(loves(Bob), left=True).draw(
    ...     path='docs/_static/biclosed/alice-loves-bob.png',
    ...     margins=(.3, 0), figsize=(5, 4))
    """
    dom: Ty
    cod: Ty
    freevars: list[Variable]
    functor: ClassVar[Functor] = Functor.id(Diagram)

    def __eq__(self, other):
        return isinstance(other, TermBase)\
            and self.alpha_equal(
                other, Substitution(()), Substitution(()))

    def __hash__(self):
        return hash(self.alpha_key(Substitution(())))

    @abstractmethod
    def eval(functor: Functor = None) -> BiclosedCategory:
        """
        The evaluation of a :class:`Functor` on a term gives a morphism in its
        codomain. By default, this is the identity functor on the free biclosed
        category, i.e. terms are compiled to diagrams with constants as boxes.
        """

    @abstractmethod
    def __substitute__(self, subst: Substitution) -> Term:
        """
        Substitute the current term by a given substitution.
        Can be called as `subst(self)` as well.
        """
        raise NotImplementedError

    def draw(self, **kwargs):
        "Drawing a term by evaluating it in the free biclosed category."
        return self.eval().draw(**kwargs)

    @property
    def map_category(self):
        return self.functor.cod.map_factory

    def to_map(self, category=None):
        raise NotImplementedError

    def __call__(self, other, left=False):
        args = (other, self, left) if left else (self, other, left)
        return self.cod.application_factory(*args)

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        return self.alpha_key(left_sub) == other.alpha_key(right_sub)

    @abstractmethod
    def alpha_key(self, substitution):
        raise NotImplementedError

    @staticmethod
    def alpha_bound(cod, index):
        return cod.variable_factory(f"__bound_{index}", cod)


class Constant(TermBase):
    """
    A constant term of defined by a :class:`Diagram` with ``dom=X, cod=Y``.
    The constant has type ``Y`` if ``X`` is empty else it has type either
    ``Y << X`` if ``left=True`` else ``X >> Y``.

    Attributes:
        inside (Diagram): The diagram which defines the constant.
        left (Optional[bool]): Whether the domain comes from the left or right.
    """
    def __init__(self, name: Ty, cod: Ty, **kwargs):
        super().__init__(name, dom=self.ob(), cod=cod, **kwargs)
        self.freevars = []

    @property
    def constants(self):
        return [self]

    def __substitute__(self, subst: Substitution) -> Constant:
        return self

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.ar_map[self]

    def to_map(self, category=None):
        category = category or self.map_category
        cmap = category.from_box(self)
        assert_term_map(cmap, self, category)
        return cmap

    def __repr__(self):
        return factory_name(type(self)) + f"({self.name!r}, {self.cod!r})"

    def __str__(self):
        return f"{self.cod}({self.name!r})"

    def alpha_key(self, substitution):
        return ("constant", self.cod, self.name)


class Variable(TermBase):
    """
    A variable with a string as name and a :class:`Ty`.

    Attributes:
        name (str): The name of the variable
        cod (Ty): The type of the variable.
    """
    def __init__(self, name: str, cod: Ty):
        super().__init__(name, dom=cod, cod=cod)
        self.freevars = [self]

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.cod.id(functor(self.cod))

    def to_map(self, category=None):
        category = category or self.map_category
        cmap = category.id(self.cod)
        assert_term_map(cmap, self, category)
        return cmap

    def __substitute__(self, subst: Substitution) -> Term:
        return subst.lookup(self)

    @property
    def constants(self):
        return []

    __repr__ = Constant.__repr__

    def alpha_key(self, substitution):
        image = substitution(self)
        return ("free", self.cod, self.name) if image is self\
            else ("bound", image.cod, image.name)

    def same_variable(self, other):
        return isinstance(other, Variable)\
            and (self.cod, self.name) == (other.cod, other.name)


class Application(TermBase):
    """
    The application either ``func(args)`` of a term ``func`` of type ``Y << X``
    to a term ``args`` of type ``X`` or ``args(func, left=True)`` of a term
    ``args`` of type ``X`` fed as input to a term ``func`` of type ``X >> Y``.

    Attributes:
        func (Term): The function being applied.
        args (Term): The arguments to which the function is applied.
        left (bool): Whether the argument comes in from the left or right.
    """
    def __init__(self, func: Term, args: Term, left: bool = False):
        assert_isinstance(func, TermBase)
        assert_isinstance(args, TermBase)
        if not func.cod.is_exp:
            raise TypeError
        self.func, self.args, self.left = func, args, left
        if self.func.cod.exponent != self.args.cod:
            raise ValueError(
                f"Expected {self.func.cod.exponent}, got {self.args.cod}")
        cod, fname, xname = func.cod.base, str(func), str(args)
        name = f"{xname}({fname}, left=True)" if left else f"{fname}({xname})"
        dom = self.__check_dom__(func, args, left)
        super().__init__(name, dom, cod)

    def __check_dom__(self, func, args, left):
        assert_isinstance(func.cod.inside[0], Under if left else Over)
        if set(func.freevars).intersection(args.freevars):
            raise ValueError("Expected disjoint free variables.")
        self.freevars = func.freevars + args.freevars if self.left\
            else args.freevars + func.freevars
        return args.dom @ func.dom if left else func.dom @ args.dom

    def __substitute__(self, subst: Substitution) -> Term:
        return type(self)(subst(self.func), subst(self.args), self.left)

    def eval(self, functor=None):
        functor = functor or self.functor
        func = self.func.eval(functor=functor)
        args = self.args.eval(functor=functor)
        base, exponent = self.func.cod.base, self.func.cod.exponent
        ev = functor.cod.ev(
            functor(base), functor(exponent), left=not self.left)
        return args @ func >> ev if self.left else func @ args >> ev

    def to_map(self, category=None):
        category = category or self.map_category
        if getattr(self, "overlap", ()):
            raise AxiomError(messages.NON_AFFINE_TERM(*self.overlap))
        func_map = self.func.to_map(category)
        args_map = self.args.to_map(category)
        app = category.category.eval_factory(
            self.func.cod, left=not self.left)
        cmap = (args_map @ func_map if self.left else func_map @ args_map)\
            >> category.from_box(app)
        assert_term_map(cmap, self, category)
        return cmap

    def __repr__(self):
        func, args = repr(self.func), repr(self.args)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({func}, {args}{left})"

    @property
    def constants(self):
        return self.args.constants + self.func.constants if self.left\
            else self.func.constants + self.args.constants

    def alpha_key(self, substitution):
        return (
            "application",
            self.left,
            self.func.alpha_key(substitution),
            self.args.alpha_key(substitution))


class Abstraction(TermBase):
    var: Variable
    body: Term
    left: bool = False

    def __init__(self, var: Variable, body: Term, left: bool = False):
        self.var, self.body, self.left = var, body, left
        left_str = ", left=True" if left else ""
        name = f"{var.cod}(lambda {var.name}{left_str}: {body})"
        cod = var.cod >> body.cod if left else body.cod << var.cod
        dom = self.__check_dom__()
        super().__init__(name, dom, cod)

    def __check_dom__(self):
        body_freevars = self.body.freevars
        if body_freevars.count(self.var) != 1:
            raise ValueError("Expected variable to occur exactly once.")
        index = body_freevars.index(self.var)
        if self.left and index != 0:
            raise ValueError("Expected abstraction of left-most variable.")
        if not self.left and index != len(body_freevars) - 1:
            raise ValueError("Expected abstraction of right-most variable.")
        self.freevars = body_freevars[1:] if self.left else body_freevars[:-1]
        return self.body.dom[1:] if self.left else self.body.dom[:-1]

    def eval(self, functor=None):
        return (functor or self.functor)(self.body.curry(left=not self.left))

    def to_map(self, category=None):
        category = category or self.map_category
        body_map = self.body.to_map(category)
        matches = [
            index for index, variable in enumerate(self.body.freevars)
            if variable == self.var]
        if len(matches) == 0:
            raise AxiomError(messages.NON_RELEVANT_TERM.format(var=self.var))
        if len(matches) > 1:
            raise AxiomError(messages.NON_AFFINE_TERM(self.var))
        index, = matches
        lam = category.category.coeval_factory(
            self.cod, left=not self.left)
        cmap = body_map.plug_input(
            index, lam, self.cod, root_index=int(self.left))
        assert_term_map(cmap, self, category)
        return cmap

    def __substitute__(self, subst: Substitution) -> Term:
        inner_subst = subst.without((self.var, ))
        return type(self)(
            self.var, inner_subst(self.body), left=self.left)

    def __repr__(self):
        var, body = repr(self.var), repr(self.body)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({var}, {body}{left})"

    @property
    def constants(self):
        return self.body.constants

    def alpha_key(self, substitution):
        variable = self.alpha_bound(self.var.cod, len(substitution))
        return (
            "abstraction",
            self.left,
            self.var.cod,
            self.body.alpha_key(
                substitution.extend(((self.var, variable), ))))


type Term = Constant | Variable | Application | Abstraction


def assert_term_map(cmap, term, category: type[CMap] | None = None):
    category = category or term.map_category
    if cmap.dom != category.ob().tensor(
            *(variable.cod for variable in term.freevars)):
        raise ValueError
    if cmap.cod != term.cod:
        raise ValueError
    if not term.constants and any(len(cycle) != 3 for cycle in cmap.node_cycles):
        raise ValueError


@dataclass
class Substitution[V: Variable, T: Term]:
    inside: dict[V, T] | tuple[tuple[V, T], ...]

    def __len__(self):
        return len(tuple(self.items()))

    def items(self):
        return self.inside.items() if hasattr(self.inside, "items")\
            else self.inside

    def lookup(self, v: V) -> T:
        try:
            # lookup without resorting to `TermBase.__hash__`
            # because hashing depends on substitutions and would
            # otherwise trigger an infinite loop
            return next(t for x, t in self.items() if x == v)
        except StopIteration:
            return v

    def extend(self, inside: dict[V, T] | tuple[tuple[V, T], ...]) -> Self:
        items = inside.items() if hasattr(inside, "items") else inside
        return type(self)(tuple(self.items()) + tuple(items))

    def without(self, variables: Iterable[V]) -> Self:
        return type(self)(tuple(
            (k, v) for k, v in self.items()
            if all(not k.same_variable(variable)
                   for variable in variables)))

    def __call__(self, term: T) -> T:
        assert_isinstance(term, TermBase)
        return term.__substitute__(self)


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Ty.over_factory, Ty.under_factory, Ty.exp_factory = Over, Under, Exp
