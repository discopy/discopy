# -*- coding: utf-8 -*-

"""
The Kleisli category of a monad on the category of Python functions.

Given a :class:`~discopy.kleisli.monad.Monad` ``M``, a channel from ``X`` to
``Y`` is a :class:`~discopy.python.function.Function` from ``X`` to
``M(Y)``. Channels compose by *binding*: the composite of ``f : X -> M(Y)``
and ``g : Y -> M(Z)`` first applies ``f``, then the functorial action of
``M`` on ``g``, then the monad's multiplication

.. math::
    f \\mathbin{;} g = f \\mathbin{;} M(g) \\mathbin{;} \\mu_Z
    \\;:\\; X \\to M(Z)

with the identity on ``X`` given by the unit :math:`\\eta_X : X \\to M(X)`.
The unit and associativity laws of the monad are exactly what is needed for
this to define a category, called the Kleisli category of ``M``.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Channel
"""
from __future__ import annotations

from discopy.abc import Category, NamedGeneric
from discopy.kleisli.monad import Monad
from discopy.python.function import Function
from discopy.utils import assert_iscomposable, assert_isinstance, factory


@factory
class Channel(Category, NamedGeneric['monad']):
    """
    A channel is a morphism in the Kleisli category of a monad ``M``, i.e. a
    :class:`~discopy.python.function.Function` from ``X`` to ``M(Y)``.

    Parameters:
        inside : The underlying function, from ``X`` to ``M(Y)``.
        dom : The domain type ``X``.
        cod : The codomain type ``Y``.

    Note
    ----
    The monad ``M`` is fixed by specialising the class with ``Channel[M]``,
    e.g. :code:`Channel[Maybe]` is the Kleisli category of the maybe monad,
    i.e. Python functions that may fail.

    Example
    -------
    >>> from discopy.kleisli.monad import Maybe
    >>> Safe = Channel[Maybe]
    >>> half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    >>> increment = Safe(lambda x: x + 1, int, int)
    >>> assert (half >> increment)(4) == 3
    >>> assert (half >> increment)(3) is None
    >>> identity = Safe.id(int)
    >>> assert (identity >> half)(4) == half(4) == (half >> identity)(4)
    """
    ob = type
    monad: Monad = None

    def __init__(self, inside: callable, dom: type, cod: type):
        monad = type(self).monad
        self.inside = inside if isinstance(inside, Function)\
            else Function(inside, dom, monad(cod))
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom: type) -> Channel:
        """
        The identity channel on a type ``dom``, given by the monad's unit.

        Parameters:
            dom : The type on which to take the identity.
        """
        return cls(cls.monad.unit(dom), dom, dom)

    def then(self, other: Channel) -> Channel:
        """
        The Kleisli composition of two channels, called with :code:`>>`.

        Parameters:
            other : The other channel to compose in sequence.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        monad = type(self).monad
        bind = self.inside\
            >> monad.functor(other.inside) >> monad.mult(other.cod)
        return type(self)(bind, self.dom, other.cod)

    def __call__(self, x):
        return self.inside(x)

    def __repr__(self):
        return f"Channel[{self.monad}]"\
            f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"
