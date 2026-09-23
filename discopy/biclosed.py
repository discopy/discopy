# -*- coding: utf-8 -*-

"""
The free biclosed monoidal category, i.e. with left and right exponentials.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
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
    CMap
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Sampler
    Renamed
    Canonical
    AlphaEquation

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, z << y), Box('g', x @ y, z), Box('h', y, x >> z)

>>> Equation(f.uncurry(left=True).curry(left=True), f).draw(
...     doctest='docs/_static/biclosed/curry-left.svg', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-left.svg
    :align: center

>>> Equation(h.uncurry(left=False).curry(left=False), h).draw(
...     doctest='docs/_static/biclosed/curry-right.svg', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-right.svg
    :align: center

>>> Equation(
...     g.curry(left=True).uncurry(left=True), g,
...     g.curry(left=False).uncurry(left=False)).draw(
...         doctest='docs/_static/biclosed/uncurry.svg')

.. image:: /_static/biclosed/uncurry.svg
    :align: center

Compact currying
----------------

:meth:`Diagram.to_compact` bends curry bubbles into coevaluation and feedback,
which lands in :class:`CMap` as a biclosed category has no trace.

>>> g.curry(left=True).uncurry(left=True).to_compact().draw(show=False,
...     doctest="docs/_static/cmap/biclosed-curry-left.dot")

.. graphviz:: /_static/cmap/biclosed-curry-left.dot
    :align: center

>>> g.curry(left=False).uncurry(left=False).to_compact().draw(show=False,
...     doctest="docs/_static/cmap/biclosed-curry-right.dot")

.. graphviz:: /_static/cmap/biclosed-curry-right.dot
    :align: center

"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from inspect import signature
from itertools import count
from typing import (
    TYPE_CHECKING, Callable, ClassVar, Iterator, Self, Sequence)

from discopy import monoidal, cmap
from discopy.abc import BiclosedCategory
from discopy.axioms import GENERATORS, Equivalence, Testable, axiom
from discopy.drawing import Drawing
from discopy.cat import factory
from discopy.utils import (
    NamedGeneric,
    assert_isatomic,
    assert_isinstance,
    deprecated_alias,
    factory_name,
    from_tree,
)

if TYPE_CHECKING:
    from hypothesis import strategies as st


@factory
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
        return self.ar(self.exp_factory(self, other))

    def over(self, other: Ty) -> Ty:
        return self.ar(self.over_factory(self, other))

    def under(self, other: Ty) -> Ty:
        return self.ar(self.under_factory(self, other))

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

    @classmethod
    def strategy(cls, *, max_leaves=3):
        """
        Generate atomic types and their exponentials, ``max_leaves`` atoms
        at most.
        """
        from hypothesis import strategies as st

        def exponentials(types):
            return st.builds(lambda x, y: x << y, types, types)\
                | st.builds(lambda x, y: x >> y, types, types)

        return st.recursive(
            super().strategy(), exponentials, max_leaves=max_leaves)

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


class Wire(monoidal.Wire):
    """
    A biclosed object is a self-dagger :class:`monoidal.Wire`, i.e. its left
    and right colours always match. Exponentials do not interact meaningfully
    with colours, so for now we assume everything is transparent.
    """
    def dagger(self) -> Wire:
        return self


class Exp(Wire):
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

    ob = Ty

    def curry(self, n=1, left=True) -> Diagram:
        """
        Wrapper around :class:`Curry` called by :class:`Functor`.

        Parameters:
            n : The number of atomic types to curry.
            left : Whether to curry on the left, i.e. into :class:`Over`,
                or on the right, i.e. into :class:`Under`.
        """
        return self.curry_factory(self, n, left)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> Eval:
        """
        Wrapper around :class:`Eval` called by :class:`Functor`.

        Parameters:
            base : The base of the exponential type to evaluate.
            exponent : The exponent of the exponential type to evaluate.
            left : Whether to evaluate on the left, i.e. from :class:`Over`,
                or on the right, i.e. from :class:`Under`.
        """
        return cls.eval_factory(
            base << exponent if left else exponent >> base)

    def to_compact(self) -> CMap:
        """
        Bend curry bubbles into coevaluation and feedback, which lands in
        :class:`CMap` as a biclosed category has no trace, see
        :meth:`discopy.cmap.CMap.to_compact`.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f.to_map() >> CMap.ev(z, y).dagger()).trace()
        """
        return self.to_map().to_compact()

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

    Note
    ----
    This is not the unit of the adjunction, which sends ``z`` to
    ``(z @ x) << x``, but the transpose of :class:`Eval`, which needs the
    exponent to be dualisable: a biclosed category has no such morphism
    unless its exponential is read at a reflexive object, see `Zeilberger
    (2016) <https://arxiv.org/abs/1512.06751>`_. It is used by
    :meth:`Curry.to_drawing` and :meth:`Diagram.to_compact`.
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

    Example
    -------
    >>> x, y, z = map(Ty, "xyz")
    >>> print(Curry(Box('f', x @ y, z)))
    Curry(f, 1, False)
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

    def __str__(self):
        return self.name

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


Id = Diagram.id
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.sum_factory = Sum


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


CMap = cmap.CMap[Diagram]


Diagram.functor_factory = Functor


class TermBase(Box, Equivalence):
    """
    A term in the internal language of biclosed categories, an
    :class:`discopy.axioms.Equivalence` up to the names of its bound
    variables, see :meth:`alpha_eq`.

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
    ...     doctest='docs/_static/biclosed/alice-loves-bob.svg',
    ...     margins=(.3, 0), figsize=(5, 4))
    """
    dom: Ty
    cod: Ty
    freevars: list[Variable]
    functor: ClassVar[Functor] = Functor.id(Diagram)

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

    def __call__(self, other, left=False):
        args = (other, self, left) if left else (self, other, left)
        return self.cod.application_factory(*args)

    def alpha_eq(self, *others: Term) -> bool:
        """
        Whether the terms are all alpha-equivalent to this one, i.e. equal up
        to the names of their bound variables, in one pass over the terms.

        Free variables are compared by name and bound ones by the depth of
        the binder they refer to, see :meth:`alpha_eq_under`.
        Alpha-equivalent terms evaluate to the same diagram, the converse
        does not hold.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> f, x, y = (Y << X)("f"), Variable("x", X), Variable("y", X)
        >>> assert X(lambda x: f(x)).alpha_eq(
        ...     X(lambda y: f(y)), X(lambda z: f(z)))
        >>> assert f(x).eval() == f(y).eval() and not f(x).alpha_eq(f(y))
        >>> h = ((Y << X) << X)("h")
        >>> assert X(lambda x: X(lambda y: h(x)(y))).alpha_eq(
        ...     X(lambda y: X(lambda x: h(y)(x))))
        >>> assert not X(lambda x: x).alpha_eq(Y(lambda y: y))
        """
        scopes = [{} for _ in (self, *others)]
        return self.alpha_eq_under(scopes, list(others))

    def alpha_eq_under(  # pylint: disable=unused-argument  # a constant
            self, scopes: list[dict[Variable, int]], others: list[Term],
            depth: int = 0) -> bool:
        """
        Whether the terms, this one and the ``others``, are alpha-equivalent
        under a scope for each of them, mapping each variable bound above it
        to the depth of its binder, i.e. its de Bruijn level, the same in
        every term: the terms are alpha-equivalent when they are equal with
        their bound variables read as levels, which is checked without
        building anything. A :class:`Variable` compares its levels, itself
        when its scope says nothing of it; a term that binds nothing and
        refers to no binder, i.e. a :class:`Constant`, is alpha-equivalent
        to its equals, reading neither ``scopes`` nor ``depth``; every
        other term former recurses into its subterms, with the arguments
        spelt out rather than unpacked, so that the recursion runs in
        Python frames, as deep as the recursion limit allows, rather than
        through C.

        Entering a binder extends every scope in place with the variable it
        binds at the current ``depth`` and leaving it restores them, so that
        comparing the terms is one pass over them, linear in their size, and
        the scopes read the same after the call.

        Example
        -------
        >>> X = Ty("X")
        >>> x, y, c = Variable("x", X), Variable("y", X), X("c")
        >>> assert x.alpha_eq_under([{x: 0}, {y: 0}], [y])
        >>> assert not x.alpha_eq_under([{x: 0}, {y: 1}], [y])
        >>> assert not x.alpha_eq_under([{x: 0}, {}], [y])
        >>> assert c.alpha_eq_under([{x: 0}, {}], [c])
        >>> assert not c.alpha_eq_under([{}, {}], [X("d")])
        """
        return all(other == self for other in others)

    @classmethod
    def generate(cls, cod: Ty, choices: Sequence[int], types: Sequence[Ty],
                 letters: Sequence[str]) -> Term:
        """
        Build a term of type ``cod`` from a sequence of ``choices``, one per
        node of the term, taking leaves once it runs out; the exponents of
        its applications are drawn from ``types`` and the variable bound at
        each level of binders is named by the letter at that level, so that
        the same choices under other letters give an alpha-equivalent term.
        Free variables are named ``v0, v1, ...`` and constants ``c0, c1, ...``.
        The term is planar and linear, as any term of a closed category is
        too, see :class:`Sampler` for the procedure.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> term = TermBase.generate(Y << X, [8, 0, 0], [X], "xy")
        >>> print(term)
        X(lambda x0: (Y << X)('c0')(x0))
        >>> assert term.alpha_eq(
        ...     TermBase.generate(Y << X, [8, 0, 0], [X], "z"))
        """
        return Sampler(cls, iter(choices), types, letters).term(cod)

    @classmethod
    def choices(cls) -> st.SearchStrategy[list[int]]:
        """ Generate the choices of :meth:`generate`, one per node. """
        from hypothesis import strategies as st

        return st.lists(st.integers(min_value=0, max_value=11), max_size=12)

    @classmethod
    def shapes(cls, *, types=None, cod=None) -> st.SearchStrategy[tuple]:
        """
        Generate the arguments of :meth:`generate` but its letters: a type,
        the choices and the types the exponents are drawn from, so that one
        shape under several namings gives :class:`Renamed` terms.
        """
        from hypothesis import strategies as st

        types = cls.ob.strategy() if types is None else types
        cods = types if cod is None else st.just(cod)
        return st.tuples(
            cods,
            cls.choices(),
            st.lists(types, min_size=1, max_size=3))

    @classmethod
    def namings(cls) -> st.SearchStrategy[list[str]]:
        """ Generate the letters naming the bound variables, one per level. """
        from hypothesis import strategies as st

        return st.lists(st.sampled_from("xyz"), min_size=1, max_size=3)

    @classmethod
    def strategy(cls, **params) -> st.SearchStrategy[Term]:
        """
        Generate terms, one :meth:`generate` call per shape and naming.

        Parameters:
            params : Passed to :meth:`shapes`.
        """
        from hypothesis import strategies as st

        return st.builds(
            lambda shape, letters: cls.generate(*shape, letters),
            cls.shapes(**params), cls.namings())

    @classmethod
    def related(cls, **params) -> st.SearchStrategy[tuple]:
        """
        Generate a term, its canonical form and a second term of the same
        type, the first again under another naming or another shape, see
        :class:`Canonical`: the first two alpha-equivalent by construction,
        the third alpha-equivalent to them or not.

        Parameters:
            params : Passed to :meth:`shapes`.
        """
        from hypothesis import strategies as st

        return st.builds(
            lambda terms: (terms[0], terms[2], terms[1]),
            Canonical[cls].strategy(**params))

    serialisation = Testable.serialisation.failing(
        "A term does not read back from its tree, see #692.")

    @axiom
    def alpha_renaming(cls, terms: Renamed[Self]) -> Equation:
        """ A term is alpha-equivalent to its renamings, however many. """
        return AlphaEquation(*terms)

    @axiom
    def alpha_application(cls, terms: Renamed[Self]) -> Equation:
        """
        Alpha-equivalence is a congruence with respect to application: a
        function applied to alpha-equivalent arguments gives alpha-equivalent
        terms, and so do alpha-equivalent functions applied to an argument.
        """
        x = cls.ob(GENERATORS[0])
        f, a = ((x << x) << terms[0].cod)("f"), x("a")
        return AlphaEquation(*(f(term)(a) for term in terms))

    @axiom
    def alpha_abstraction(cls, terms: Renamed[Self]) -> Equation:
        """
        Alpha-equivalence is a congruence with respect to abstraction:
        binding a variable in alpha-equivalent bodies gives alpha-equivalent
        terms.
        """
        x = cls.ob(GENERATORS[0])
        f, w = ((x << x) << terms[0].cod)("f"), cls.ob.variable_factory("w", x)
        return AlphaEquation(*(
            cls.ob.abstraction_factory(w, f(term)(w), False)
            for term in terms))

    @axiom
    def alpha_soundness(cls, terms: Renamed[Self]) -> Equation:
        """ Alpha-equivalent terms evaluate to the same diagram. """
        return Equation(*(term.eval() for term in terms))

    @axiom
    def alpha_completeness(cls, terms: Canonical[Self]) -> Equation:
        """
        Alpha-equivalence is decided by the canonical naming of the bound
        variables: two terms are alpha-equivalent exactly when their
        canonical forms are equal, see :class:`Canonical`. The other laws
        hold of a relation that says yes too often; this one fails when
        terms that are not alpha-equivalent are.
        """
        first, second, *canonical = terms
        return Equation(first.alpha_eq(second), canonical[0] == canonical[1])


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

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.ar_map[self]

    def __repr__(self):
        return factory_name(type(self)) + f"({self.name!r}, {self.cod!r})"

    def __str__(self):
        return f"{self.cod!s}({self.name!r})"


class Variable(TermBase):
    """
    A variable with a string as name and an atomic :class:`Ty`.

    A variable stands for exactly one wire in the internal language of a
    (bi)closed category, the way a lambda term binds one variable at a
    time: the abstraction machinery indexes contexts and free variables by
    variable, counting on that index to coincide with a wire index.

    Attributes:
        name (str): The name of the variable
        cod (Ty): The atomic type of the variable.
    """
    def __init__(self, name: str, cod: Ty):
        assert_isatomic(cod)
        super().__init__(name, dom=cod, cod=cod)
        self.freevars = [self]

    def eval(self, functor=None):
        functor = functor or self.functor
        return functor.cod.id(functor(self.cod))

    @property
    def constants(self):
        return []

    def alpha_eq_under(  # pylint: disable=unused-argument  # depth: binders
            self, scopes, others, depth=0):
        """ Variables alike by the level of their binder, by name if free. """
        if any(type(other) is not type(self) for other in others):
            return False
        images = [scope.get(term, term)
                  for scope, term in zip(scopes, (self, *others))]
        return all(image == images[0] for image in images[1:])

    __repr__ = Constant.__repr__


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
            raise TypeError(f"Expected {Exp}, got {type(func.cod)}")
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
        self.freevars = args.freevars + func.freevars if self.left\
            else func.freevars + args.freevars
        return args.dom @ func.dom if left else func.dom @ args.dom

    def eval(self, functor=None):
        functor = functor or self.functor
        func = self.func.eval(functor=functor)
        args = self.args.eval(functor=functor)
        base, exponent = self.func.cod.base, self.func.cod.exponent
        ev = functor.cod.ev(
            functor(base), functor(exponent), left=not self.left)
        return args @ func >> ev if self.left else func @ args >> ev

    def __repr__(self):
        func, args = repr(self.func), repr(self.args)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({func}, {args}{left})"

    @property
    def constants(self):
        return self.args.constants + self.func.constants if self.left\
            else self.func.constants + self.args.constants

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Applications on one side, functions and arguments alike. """
        if any(type(other) is not type(self) or other.left != self.left
               for other in others):
            return False
        return self.func.alpha_eq_under(
            scopes, [other.func for other in others], depth)\
            and self.args.alpha_eq_under(
                scopes, [other.args for other in others], depth)


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

    def __repr__(self):
        var, body = repr(self.var), repr(self.body)
        left = ", left=True" if self.left else ""
        return factory_name(type(self)) + f"({var}, {body}{left})"

    @property
    def constants(self):
        return self.body.constants

    def alpha_eq_under(self, scopes, others, depth=0):
        """ Binders of one type on one side, bodies alike one level down. """
        if any(type(other) is not type(self)
               or (other.left, other.var.cod) != (self.left, self.var.cod)
               for other in others):
            return False
        terms = (self, *others)
        shadowed = [scope.get(term.var) for scope, term in zip(scopes, terms)]
        for scope, term in zip(scopes, terms):
            scope[term.var] = depth
        result = self.body.alpha_eq_under(
            scopes, [other.body for other in others], depth + 1)
        for scope, term, level in zip(scopes, terms, shadowed):
            if level is None:
                del scope[term.var]
            else:
                scope[term.var] = level
        return result


type Term = Constant | Variable | Application | Abstraction


for law in (
        "unitality", "associativity", "identity_typing",
        "composition_dom_typing", "composition_cod_typing",
        "dagger_involution", "dagger_contravariance"):
    setattr(TermBase, law, getattr(Box, law).inapplicable(
        "A term is a box: the laws of its category are the diagram's."))


@dataclass
class Sampler:
    """
    Samples a term from a sequence of choices, one per node, taking leaves
    once it runs out, see :meth:`TermBase.generate`.

    The term is planar and linear: the free variables of each subterm open
    with a ``prefix`` and close with a ``suffix`` of the bound variables in
    scope, with ``extra`` free ones in between or not, which an application
    :meth:`splits` between its function and its argument and an
    :meth:`abstraction` extends with the variable it binds; a subterm that
    cannot be a leaf is a :meth:`spine`, a constant applied to the bound
    variables in order. The options of a choice are closures calling the
    method that builds the node with its arguments spelt out, so that the
    recursion runs in Python frames, as deep as the recursion limit allows:
    :func:`functools.partial`, or a call through ``*args``, goes through C
    at every level and overflows its stack a few thousand nodes deep.

    Parameters:
        category : The class of terms to sample.
        choices : The choices left, consumed one per node.
        types : The types the exponents of applications are drawn from.
        letters : The letters naming the bound variables, one per level.
        counter : The numbers of the free variables and constants built.
        linear : Whether each bound variable occurs exactly once, in the
            order of the binders, as :mod:`biclosed` requires; a term of
            :mod:`discopy.closed` may use one any number of times.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> sampler = Sampler(TermBase, iter([]), [X], "x")
    >>> print(sampler.spine(Y, (Variable("x0", X), )))
    (Y << X)('c0')(x0)
    >>> print(sampler.term(Y << X))
    (Y << X)('c1')
    """
    category: type[TermBase]
    choices: Iterator[int]
    types: Sequence[Ty]
    letters: Sequence[str]
    counter: Iterator[int] = field(default_factory=count)
    linear: bool = True

    def choose(self, options: Sequence):
        """ Pick an option by the next choice, the first when they ran out. """
        return options[next(self.choices, 0) % len(options)]

    def constant(self, cod: Ty) -> Constant:
        """ A fresh constant of a given type. """
        return self.category.ob.constant_factory(
            f"c{next(self.counter)}", cod)

    def variable(self, cod: Ty) -> Variable:
        """ A fresh free variable of a given type. """
        return self.category.ob.variable_factory(
            f"v{next(self.counter)}", cod)

    def bound(self, level: int, cod: Ty) -> Variable:
        """ The variable bound at a given level, named by its letter. """
        letter = self.letters[level % len(self.letters)]
        return self.category.ob.variable_factory(f"{letter}{level}", cod)

    def spine(self, cod: Ty, bound: tuple[Variable, ...]) -> Term:
        """ A fresh constant applied to the bound variables in order. """
        function_type = cod
        for variable in reversed(bound):
            function_type = function_type << variable.cod
        result = self.constant(function_type)
        for variable in bound:
            result = self.category.ob.application_factory(
                result, variable, False)
        return result

    @staticmethod
    def leaf(variable: Variable) -> Callable[[], Variable]:
        """ The option of a bound variable as a leaf. """
        return lambda: variable

    def leaves(self, cod: Ty, bound: tuple[Variable, ...],
               extra: bool) -> list[Callable[[], Term]]:
        """
        The leaves allowed: a constant or, if ``extra``, a free variable
        when no variable is bound, and the bound variables of the type; a
        linear term takes the only bound variable and nothing else.
        """
        result = []
        if not bound or not self.linear:
            result.append(lambda: self.constant(cod))
        if (not bound or not self.linear) and extra:
            result.append(lambda: self.variable(cod))
        if not self.linear or len(bound) == 1:
            result += [self.leaf(variable)
                       for variable in bound if variable.cod == cod]
        return result

    def splits(self, prefix: tuple, suffix: tuple, extra: bool) -> list:
        """
        The ways of splitting the constraints between two subterms in
        sequence: at a bound variable of the prefix, among the extras or at
        a bound variable of the suffix, each a pair of constraints; a term
        that need not be linear passes every bound variable to both.
        """
        if not self.linear:
            return [((prefix, suffix, extra), (prefix, suffix, extra))]
        result = [((prefix[:i], (), False), (prefix[i:], suffix, extra))
                  for i in range(len(prefix) + 1)]
        result += [((prefix, (), True), ((), suffix, True))] if extra else []
        result += [((prefix, suffix[:j], extra), (suffix[j:], (), False))
                   for j in range(len(suffix) + 1)]
        return result

    def application(self, cod: Ty, first: tuple, second: tuple, left: bool,
                    level: int) -> Application:
        """
        A function applied to an argument of an exponent drawn from the
        types, ``first`` the constraints of whichever comes first in the
        free variables: the function, or the argument if ``left``.
        """
        exponent = self.choose(self.types)
        if left:
            args = self.term(exponent, first, level)
            func = self.term(exponent >> cod, second, level)
            return self.category.ob.application_factory(func, args, True)
        func = self.term(cod << exponent, first, level)
        args = self.term(exponent, second, level)
        return self.category.ob.application_factory(func, args, False)

    def applications(self, cod: Ty, prefix: tuple, suffix: tuple,
                     extra: bool, level: int) -> list[Callable[[], Term]]:
        """ The applications allowed, one per split and side. """
        return [self.split(cod, first, second, left, level)
                for first, second in self.splits(prefix, suffix, extra)
                for left in (False, True)]

    def split(self, cod: Ty, first: tuple, second: tuple, left: bool,
              level: int) -> Callable[[], Application]:
        """ The option of an application with its constraints split. """
        return lambda: self.application(cod, first, second, left, level)

    def abstraction(self, cod: Ty, prefix: tuple, suffix: tuple, extra: bool,
                    level: int) -> Abstraction:
        """
        The abstraction of the variable bound at this level, first in the
        free variables of the body when the type is a left exponential and
        last otherwise.
        """
        left = cod.is_under
        var = self.bound(level, cod.exponent)
        opening, closing = ((var, *prefix), suffix) if left\
            else (prefix, (*suffix, var))
        body = self.term(cod.base, (opening, closing, extra), level + 1)
        return self.category.ob.abstraction_factory(var, body, left)

    def term(self, cod: Ty, constraints: tuple = ((), (), True),
             level: int = 0) -> Term:
        """
        A term of a given type under the constraints, i.e. the prefix, the
        suffix and whether extra free variables are allowed: a leaf when the
        choices ran out, or the spine when none fits, else the option the
        next choice picks among the leaves, the applications and the
        abstraction if the type is an exponential.
        """
        prefix, suffix, extra = constraints
        bound = prefix + suffix
        options = self.leaves(cod, bound, extra)
        choice = next(self.choices, None)
        if choice is None:
            return options[0]() if options else self.spine(cod, bound)
        options += self.applications(cod, prefix, suffix, extra, level)
        if cod.is_exp:
            options.append(
                lambda: self.abstraction(cod, prefix, suffix, extra, level))
        return options[choice % len(options)]()


class Renamed(Testable, NamedGeneric["factory"], tuple):
    """
    Alpha-equivalent terms of the ``factory``: one shape generated under two
    or more namings of its bound variables, see :meth:`TermBase.generate`.

    Example
    -------
    >>> from hypothesis import find
    >>> terms = find(
    ...     Renamed[TermBase].strategy(), lambda terms: len(terms) == 3)
    >>> assert terms[0].alpha_eq(*terms[1:])
    """
    @classmethod
    def strategy(cls, **params) -> st.SearchStrategy[Renamed]:
        """
        Generate a shape and at least two namings of it.

        Parameters:
            params : Passed to :meth:`TermBase.shapes`.
        """
        from hypothesis import strategies as st

        category = cls.factory
        return st.builds(
            lambda shape, namings: cls(
                category.generate(*shape, letters) for letters in namings),
            category.shapes(**params),
            st.lists(category.namings(), min_size=2, max_size=3))


class Canonical(Testable, NamedGeneric["factory"], tuple):
    """
    Two terms of the ``factory`` followed by their canonical forms: two
    shapes of one type, the second the first's or another, each generated
    under its own naming and then under the one naming ``"x"``, which names
    every bound variable by its level, so that the two terms are
    alpha-equivalent exactly when their canonical forms are equal. Where
    :class:`Renamed` draws terms alpha-equivalent by construction, this
    decides the alpha-equivalence of any pair, see
    :meth:`TermBase.alpha_completeness`.

    Example
    -------
    >>> from hypothesis import find
    >>> first, second, *canonical = find(
    ...     Canonical[TermBase].strategy(),
    ...     lambda terms: terms[0] != terms[1] and terms[2] == terms[3])
    >>> assert first.alpha_eq(second)
    """
    @classmethod
    def build(cls, shape: tuple, others: list[int] | None,
              namings: tuple[Sequence[str], Sequence[str]]) -> Canonical:
        """
        The terms of a shape and of another, the same when ``others`` is
        ``None``, under their namings and then under the canonical one.
        """
        cod, choices, types = shape
        shapes = (choices, choices if others is None else others)
        return cls(
            cls.factory.generate(cod, each, types, letters)
            for each, letters in zip(2 * shapes, (*namings, "x", "x")))

    @classmethod
    def strategy(cls, **params) -> st.SearchStrategy[Canonical]:
        """
        Generate a shape, its choices again or others, and two namings.

        Parameters:
            params : Passed to :meth:`TermBase.shapes`.
        """
        from hypothesis import strategies as st

        category = cls.factory
        return st.builds(
            cls.build, category.shapes(**params),
            st.none() | category.choices(),
            st.tuples(category.namings(), category.namings()))


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Ty.over_factory, Ty.under_factory, Ty.exp_factory = Over, Under, Exp


class Equation(monoidal.Equation):
    """ The :class:`monoidal.Equation` of biclosed diagrams. """


class AlphaEquation(Equation):
    """
    An :class:`Equation` between terms which holds when they are
    alpha-equivalent, see :meth:`TermBase.alpha_eq`.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> f = (Y << X)("f")
    >>> assert AlphaEquation(X(lambda x: f(x)), X(lambda y: f(y)))
    >>> assert not AlphaEquation(f(Variable("x", X)), f(Variable("y", X)))
    """
    def __bool__(self):
        term, *others = self.terms
        return term.alpha_eq(*others)


TermBase.equivalence_factory = AlphaEquation

__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})
