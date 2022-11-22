# -*- coding: utf-8 -*-

"""
The free closed monoidal category, i.e. with exponential objects.

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
    Curry
    Category
    Functor

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', y, z << x), Box('g', y, z >> x)

>>> from discopy.python import Function
>>> F = Functor(
...     ob={x: complex, y: bool, z: float},
...     ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
...         g: lambda y: lambda z: z + 1j if y else -1j},
...     cod=Category(tuple[type, ...], Function))

>>> assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
>>> assert F(g.uncurry(left=False).curry(left=False))(True)(1.2) == F(g)(True)(1.2)
"""

from __future__ import annotations

from discopy import cat, monoidal, rigid, messages
from discopy.cat import Category, factory
from discopy.utils import (
    factory_name,
    BinaryBoxConstructor,
    assert_isinstance,
    from_tree,
)


@factory
class Ty(monoidal.Ty):
    """
    A closed type is a monoidal type that can be exponentiated.

    Parameters:
        inside (Ty) : The objects inside the type.
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(other, self)

    def __repr__(self):
        return "{}({})".format(
            factory_name(type(self)), ', '.join(map(repr, self.inside)))


class Exp(Ty, cat.Ob):
    """
    A :code:`base` type to an :code:`exponent` type, called with :code:`**`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    __ambiguous_inheritance__ = (cat.Ob, )

    def __init__(self, base: Ty, exponent: Ty):
        self.base, self.exponent = base, exponent
        # TODO : replace left and right by base and exponent
        self.left, self.right =\
            (exponent, base) if isinstance(self, Under) else (base, exponent)
        super().__init__(self)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.base, self.exponent) == (other.base, other.exponent)
        if isinstance(other, Exp):
            return False  # Avoid infinite loop with Over(x, y) == Under(x, y).
        return isinstance(other, Ty) and other.inside == (self, )

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "({} ** {})".format(self.base, self.exponent)

    def __repr__(self):
        return factory_name(type(self)) + "({}, {})".format(
            repr(self.base), repr(self.exponent))

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'base': self.base.to_tree(),
            'exponent': self.exponent.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['base'], tree['exponent'])))


class Over(Exp):
    """
    An :code:`exponent` type over a :code:`base` type, called with :code:`<<`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return "({} << {})".format(self.base, self.exponent)


class Under(Exp):
    """
    A :code:`base` type under an :code:`exponent` type, called with :code:`>>`.

    Parameters:
        base : The base type.
        exponent : The exponent type.
    """
    def __str__(self):
        return "({} >> {})".format(self.exponent, self.base)


@factory
class Diagram(monoidal.Diagram):
    """
    A closed diagram is a monoidal diagram
    with :class:`Curry` and :class:`Eval` boxes.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    __ambiguous_inheritance__ = True

    def curry(self, n=1, left=True) -> Diagram:
        """
        Wrapper around :class:`Curry` called by :class:`Functor`.

        Parameters:
            n : The number of atomic types to curry.
            left : Whether to curry on the left or right.
        """
        return Curry(self, n, left)

    @staticmethod
    def eval(base: Ty, exponent: Ty, left=True) -> Eval:
        """
        Wrapper around :class:`Eval` called by :class:`Functor`.

        Parameters:
            base : The base of the exponential type to evaluate.
            exponent : The exponent of the exponential type to evaluate.
            left : Whether to evaluate on the left or right.
        """
        return Eval(base << exponent if left else exponent >> base)

    def uncurry(self: Diagram, left=True) -> Diagram:
        """
        Uncurry a closed diagram by composing it with :meth:`Diagram.eval`.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        base, exponent = self.cod.base, self.cod.exponent
        return self @ exponent >> self.eval(base, exponent, True) if left\
            else exponent @ self >> self.eval(base, exponent, False)

    @staticmethod
    def fa(left, right):
        """ Forward application. """
        return FA(left << right)

    @staticmethod
    def ba(left, right):
        """ Backward application. """
        return BA(left >> right)

    @staticmethod
    def fc(left, middle, right):
        """ Forward composition. """
        return FC(left << middle, middle << right)

    @staticmethod
    def bc(left, middle, right):
        """ Backward composition. """
        return BC(left >> middle, middle >> right)

    @staticmethod
    def fx(left, middle, right):
        """ Forward crossed composition. """
        return FX(left << middle, right >> middle)

    @staticmethod
    def bx(left, middle, right):
        """ Backward crossed composition. """
        return BX(middle << left, middle >> right)


class Box(monoidal.Box, Diagram):
    """
    A closed box is a monoidal box in a closed diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (monoidal.Box, )


class Eval(Box):
    """
    The evaluation of an exponential type.

    Parameters:
        x : The exponential type to evaluate.
    """
    def __init__(self, x: Exp):
        self.base, self.exponent = x.base, x.exponent
        self.left = isinstance(x, Over)
        dom, cod = (x @ self.exponent, self.base) if self.left\
            else (self.exponent @ x, self.base)
        super().__init__("Eval" + str(x), dom, cod)


class Curry(monoidal.Bubble, Box):
    """
    The currying of a closed diagram.

    Parameters:
        arg : The diagram to curry.
        n : The number of atomic types to curry.
        left : Whether to curry on the left or right.
    """
    def __init__(self, arg: Diagram, n=1, left=True):
        self.arg, self.n, self.left = arg, n, left
        name = "Curry({}, {}, {})".format(arg, n, left)
        if left:
            dom = arg.dom[:len(arg.dom) - n]
            cod = arg.cod << arg.dom[len(arg.dom) - n:]
        else:
            dom, cod = arg.dom[n:], arg.dom[:n] >> arg.cod
        monoidal.Bubble.__init__(
            self, arg, dom, cod, drawing_name="$\\Lambda$")
        Box.__init__(self, name, dom, cod)


def unaryBoxConstructor(attr):
    class Constructor:
        @classmethod
        def from_tree(cls, tree):
            return cls(from_tree(tree[attr]))

        def to_tree(self):
            return {
                'factory': factory_name(type(self)),
                attr: getattr(self, attr).to_tree()}
    return Constructor


class FA(unaryBoxConstructor("over"), Box):
    """ Forward application box. """
    def __init__(self, over):
        assert_isinstance(over, Over)
        self.over = over
        dom, cod = over @ over.exponent, over.base
        super().__init__("FA{}".format(over), dom, cod)

    def __repr__(self):
        return "FA({})".format(repr(self.dom[:1]))


class BA(unaryBoxConstructor("under"), Box):
    """ Backward application box. """
    def __init__(self, under):
        assert_isinstance(under, Under)
        self.under = under
        dom, cod = under.exponent @ under, under.base
        super().__init__("BA{}".format(under), dom, cod)

    def __repr__(self):
        return "BA({})".format(repr(self.dom[1:]))


class FC(BinaryBoxConstructor, Box):
    """ Forward composition box. """
    def __init__(self, left, right):
        assert_isinstance(left, Over)
        assert_isinstance(right, Over)
        if left.exponent != right.base:
            raise ValueError(messages.types_do_not_compose(left, right))
        name = "FC({}, {})".format(left, right)
        dom, cod = left @ right, left.base << right.exponent
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class BC(BinaryBoxConstructor, Box):
    """ Backward composition box. """
    def __init__(self, left, right):
        assert_isinstance(left, Under)
        assert_isinstance(right, Under)
        if left.base != right.exponent:
            raise ValueError(messages.types_do_not_compose(left, right))
        name = "BC({}, {})".format(left, right)
        dom, cod = left @ right, left.exponent >> right.base
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class FX(BinaryBoxConstructor, Box):
    """ Forward crossed composition box. """
    def __init__(self, left, right):
        assert_isinstance(left, Over)
        assert_isinstance(right, Under)
        if left.exponent != right.base:
            raise ValueError(messages.types_do_not_compose(left, right))
        name = "FX({}, {})".format(left, right)
        dom, cod = left @ right, right.exponent >> left.base
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


class BX(BinaryBoxConstructor, Box):
    """ Backward crossed composition box. """
    def __init__(self, left, right):
        assert_isinstance(left, Over)
        assert_isinstance(right, Under)
        if left.base != right.exponent:
            raise ValueError(messages.types_do_not_compose(left, right))
        name = "BX({}, {})".format(left, right)
        dom, cod = left @ right, right.base << left.exponent
        Box.__init__(self, name, dom, cod)
        BinaryBoxConstructor.__init__(self, left, right)


Diagram.over, Diagram.under, Diagram.exp\
    = map(staticmethod, (Over, Under, Exp))

Id = Diagram.id


class Category(monoidal.Category):
    """
    A closed category is a monoidal category with methods :code:`exp`
    (:code:`over` and / or :code:`under`), :code:`eval` and :code:`curry`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A closed functor is a monoidal functor
    that preserves evaluation and currying.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls):
                method = getattr(self.cod.ar, attr)
                return method(self(other.base), self(other.exponent))
        if isinstance(other, Curry):
            return self.cod.ar.curry(
                self(other.arg), len(self(other.cod.exponent)), other.left)
        if isinstance(other, Eval):
            return self.cod.ar.eval(
                self(other.base), self(other.exponent), other.left)
        if isinstance(other, FA):
            left, right = other.over.left, other.over.right
            return self.cod.ar.fa(self(left), self(right))
        if isinstance(other, BA):
            left, right = other.under.left, other.under.right
            return self.cod.ar.ba(self(left), self(right))
        for cls, method in [(FC, 'fc'), (BC, 'bc')]:
            if isinstance(other, cls):
                left = other.dom.inside[0].left
                middle = other.dom.inside[0].right
                right = other.dom.inside[1].right
                return getattr(self.cod.ar, method)(
                    self(left), self(middle), self(right))
        if isinstance(other, FX):
            left = other.dom.inside[0].left
            middle = other.dom.inside[0].right
            right = other.dom.inside[1].left
            return getattr(self.cod.ar, 'fx')(
                self(left), self(middle), self(right))
        if isinstance(other, BX):
            left = other.dom.inside[0].right
            middle = other.dom.inside[0].left
            right = other.dom.inside[1].right
            return getattr(self.cod.ar, 'bx')(
                self(left), self(middle), self(right))
        return super().__call__(other)


closed2rigid = Functor(
    ob=lambda x: rigid.Ty(x.inside[0].name),
    ar=lambda f: rigid.Box(
        f.name, closed2rigid(f.dom), closed2rigid(f.cod)),
    cod=rigid.Category())
