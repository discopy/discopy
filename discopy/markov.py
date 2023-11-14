# -*- coding: utf-8 -*-

"""
The free Markov category, i.e. a semicartesian category with a supply of
commutative comonoid, see :cite:t:`FritzLiang23`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Copy
    Category
    Functor


Axioms
------

>>> from discopy.drawing import Equation
>>> Diagram.use_hypergraph_equality = True
>>> x = Ty('x')

>>> copy, merge = Copy(x), Merge(x)
>>> unit, delete = Merge(x, n=0), Copy(x, n=0)

* Commutative monoid:

>>> unitality = Equation(unit @ x >> merge, Id(x), x @ unit >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     path="docs/_static/frobenius/monoid.png")

.. image:: /_static/frobenius/monoid.png
    :align: center

* Cocommutative comonoid:

>>> counitality = Equation(copy >> delete @ x, Id(x), copy >> x @ delete)
>>> coassociativity = Equation(copy >> copy @ x, copy >> x @ copy)
>>> cocommutativity = Equation(copy >> Swap(x, x), copy)
>>> assert counitality and coassociativity and cocommutativity
>>> Equation(counitality, coassociativity, cocommutativity, symbol='').draw(
...     path="docs/_static/frobenius/comonoid.png")

.. image:: /_static/frobenius/comonoid.png
    :align: center

* Coherence:

>>> assert Diagram.copy(x @ x, n=0) == delete @ delete
>>> assert Diagram.copy(x @ x)\\
...     == copy @ copy >> x @ Swap(x, x) @ x
>>> assert Diagram.merge(x @ x, n=0) == unit @ unit
>>> assert Diagram.merge(x @ x)\\
...     == x @ Swap(x, x) @ x >> merge @ merge

>>> Diagram.use_hypergraph_equality = False

Note
----
Equality of Markov diagrams is computed by translation to hypergraph.
Both copy and merge boxes are translated to spiders, thus when they appear
in the same diagram they automatically satisfy the :mod:`frobenius` axioms.
"""

from __future__ import annotations

from discopy import symmetric, monoidal, hypergraph
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.utils import assert_isatomic


@factory
class Diagram(symmetric.Diagram):
    """
    A Markov diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    We can create arbitrary Markov diagrams with the standard notation for
    Python functions.

    >>> x = Ty('x')
    >>> f = Box('f', x, x)

    >>> copy_then_apply = Diagram.from_callable(x, x @ x)(
    ...     lambda x: (f(x), f(x)))

    >>> @Diagram.from_callable(x, x @ x)
    ... def apply_then_copy(x):
    ...     y = f(x)
    ...     return x, x

    >>> from discopy.drawing import Equation
    >>> Equation(copy_then_apply, apply_then_copy, symbol="$\\\\neq$").draw(
    ...     path="docs/_static/markov/copy_and_apply.png")

    .. image:: /_static/markov/copy_and_apply.png
    """
    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or 1 not in (n_legs_in, n_legs_out):
            raise ValueError
        return cls.copy_factory(typ, n_legs_out) if n_legs_in == 1\
            else cls.merge_factory(typ, n_legs_in)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, 1, n, x)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.copy(x, n).dagger()

    @classmethod
    def discard(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        The discard of an atomic type :code:`x`.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)


class Box(symmetric.Box, Diagram):
    """
    A Markov box is a symmetric box in a Markov diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )


class Swap(symmetric.Swap, Box):
    """
    Symmetric swap in a Markov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, )


class Trace(symmetric.Trace, Box):
    """
    A trace in a Markov category.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """
    __ambiguous_inheritance__ = (symmetric.Trace, )


class Copy(Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        super().__init__(name=f"Copy({x}, {n})", dom=x, cod=x ** n,
                         draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.discard_factory.__new__(cls.discard_factory, x)

    def dagger(self) -> Merge:
        return Merge(self.dom, len(self.cod))


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        super().__init__(name=f"Merge({x}, {n})", dom=x ** n, cod=x,
                         draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Merge:
        return Copy(self.cod, len(self.dom))


class Discard(Copy):
    """
    The discard of an atomic type :code:`x`.

    Parameters:
        x : The type to discard.
    """
    def __init__(self, x: monoidal.Ty, *args, **kwargs):
        super().__init__(x, 0)


class Sum(symmetric.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (symmetric.Sum, )


class Category(symmetric.Category):
    """
    A Markov category is a symmetric category with a method :code:`copy`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor):
    """
    A Markov functor is a symmetric functor that preserves copies.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.

    Example
    -------

    We build a functor into python functions.

    >>> x = Ty('x')
    >>> add = Box('add', x @ x, x)
    >>> from discopy import python
    >>> F = Functor({x: int}, {add: lambda a, b: a + b},
    ...             cod=Category(python.Ty, python.Function))
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> from discopy.drawing import Equation
    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     path="docs/_static/markov/bialgebra.png")

    .. image:: /_static/markov/bialgebra.png
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        if isinstance(other, Merge):
            return self.cod.ar.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


class Hypergraph(hypergraph.Hypergraph):
    category, functor = Category, Functor

    def to_diagram(self, make_causal_first=True) -> Diagram:
        return super().to_diagram(
            make_causal_first=make_causal_first)


Diagram.hypergraph_factory = Hypergraph
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.braid_factory = Swap
Diagram.trace_factory = Trace
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Id = Diagram.id
