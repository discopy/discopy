# -*- coding: utf-8 -*-

"""
The Kleisli category of a monad with tuple as tensor.

A :class:`~discopy.kleisli.channel.Channel` only has a single wire on
either side. Here a channel from a tuple of wires ``dom`` to a tuple of
wires ``cod`` is a Python function from ``dom`` to the monadic type
``M(pack(cod))``, where :func:`pack` turns a tuple of wire types into the
single type a :class:`~discopy.kleisli.monad.Monad` already knows how to
act on: the bare wire type itself if there is only one, else a
:class:`Row` of the values on all the wires.

.. admonition:: Why not ``tuple[X, Y]``?

    The obvious choice of composite type is a genuine tuple type, e.g.
    representing two wires ``X, Y`` by ``tuple[X, Y]``. But every
    :class:`~discopy.python.function.Function` composes by
    :func:`~discopy.utils.tuplify`-ing its result and splatting it into
    the next one, so that e.g. ``f : A -> (B, C)`` composes with
    ``g : (B, C) -> D`` by unpacking ``f``'s output as ``g``'s two
    arguments. If a *single* packed value already happens to be a tuple
    (as ``M(tuple[X, Y])``'s payload would), :func:`~discopy.utils.tuplify`
    cannot tell it apart from *several* values to splat, and composition
    silently sends the wrong number of arguments downstream. :class:`Row`
    is not a tuple, so it is never mistaken for one.

Whiskering a channel by a plain tuple of types on the left or right needs
no extra structure: the untouched wires are simply carried along by
functoriality, i.e. by the *strength* of the monad

.. math::
    \\mathrm{st}_L : X \\times M(Y) \\to M(X \\times Y)
    \\qquad\\qquad
    \\mathrm{st}_R : M(X) \\times Y \\to M(X \\times Y)

which any monad on the category of Python functions has pointwise, by
pairing a plain value with the result of a functorial action. The tensor of
two channels ``f : A -> M(B)`` and ``g : C -> M(D)`` is then given, as
everywhere else in DisCoPy, by composition and whiskering

.. math::
    f \\otimes g = (f \\otimes \\mathrm{id}_C) \\mathbin{;}
    (\\mathrm{id}_B \\otimes g) \\;:\\; A \\times C \\to M(B \\times D)

biased so that the effects of ``f`` run before those of ``g``. This makes
the Kleisli category a *premonoidal* category: unlike an ordinary monoidal
category, tensoring is not required to be a bifunctor, i.e. ``f @ g`` need
not agree with the differently biased composite that runs ``g`` first --
the two coincide for every ``f, g`` if and only if the monad is
*commutative*, see e.g. `the nLab <https://ncatlab.org/nlab/show/commutative
+monad>`_ and the tests in ``test/kleisli.py``.

A channel also comes with a canonical comonoid structure, i.e. methods
:meth:`Channel.copy` and :meth:`Channel.discard`, given by duplicating (or
discarding) the input before embedding it with the monad's unit -- this
makes the Kleisli category a copy-discard category.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Row
    Channel
"""
from __future__ import annotations

from discopy.abc import Category, NamedGeneric
from discopy.kleisli import channel
from discopy.kleisli.monad import Monad
from discopy.python import function
from discopy.python.multiplicative import Ty
from discopy.utils import (
    assert_isinstance, assert_iscomposable, factory, factory_name, tuplify)


class Row:
    """
    A row of values packed into one, standing for the values on a tuple of
    wires, see the module docstring for why this needs to be its own class
    rather than a plain :class:`tuple`.

    Example
    -------
    >>> assert Row(1, 2) == eval(repr(Row(1, 2))) == Row(1, 2)
    >>> assert Row(1, 2) != (1, 2)
    """
    def __init__(self, *values):
        self.values = values

    def __eq__(self, other):
        return isinstance(other, Row) and self.values == other.values

    def __hash__(self):
        return hash(self.values)

    def __repr__(self):
        return f"Row{self.values!r}"


def pack(ty: Ty) -> type:
    """
    The composite Python type standing for a tuple of wires ``ty``, i.e.
    the single type a :class:`~discopy.kleisli.monad.Monad` acts on.

    Parameters:
        ty : The tuple of wire types to pack.

    Example
    -------
    >>> assert pack(()) is Row
    >>> assert pack((int, )) is int
    >>> assert pack((int, str)) is Row
    """
    ty = tuplify(ty)
    return ty[0] if len(ty) == 1 else Row


def pack_value(xs: tuple):
    """
    Turn a tuple of values on some wires into the value of :func:`pack` of
    their types.

    Parameters:
        xs : The tuple of values to pack.
    """
    return xs[0] if len(xs) == 1 else Row(*xs)


def unpack_value(value) -> tuple:
    """
    Turn a value of a packed type back into a tuple of values, inverse to
    :func:`pack_value`.

    Parameters:
        value : The value to unpack.
    """
    return value.values if isinstance(value, Row) else (value, )


@factory
class Channel(Category, NamedGeneric['monad']):
    """
    A channel is a morphism in the Kleisli category of a monad ``M`` with
    tuple as tensor, i.e. a Python function from a tuple of wires ``dom``
    to the packed monadic type ``M(pack(cod))``.

    Parameters:
        inside : The underlying function, from ``dom`` to ``M(pack(cod))``.
        dom : The domain, i.e. a tuple of wire types.
        cod : The codomain, i.e. a tuple of wire types.

    Note
    ----
    Composition and identities are inherited from the single-wire
    :class:`~discopy.kleisli.channel.Channel` applied to the packed domain
    and codomain, see :func:`pack`.

    Example
    -------
    >>> from discopy.kleisli.monad import Maybe, Powerset
    >>> Safe = Channel[Maybe]
    >>> half = Safe(lambda x: x // 2 if x % 2 == 0 else None, int, int)
    >>> increment = Safe(lambda x: x + 1, int, int)
    >>> parallel = half @ increment
    >>> assert parallel(4, 10) == (2, 11)
    >>> assert parallel(3, 10) is None

    The comonoid structure makes copies and discards the input:

    >>> Nondet = Channel[Powerset]
    >>> copy = Nondet.copy((int, ))
    >>> assert copy(3) == frozenset({(3, 3)})
    >>> discard = Nondet.discard((int, ))
    >>> assert discard(3) == frozenset({()})
    """
    ob = Ty
    monad: Monad = None

    def __init__(self, inside: callable, dom: Ty, cod: Ty):
        dom, cod = tuplify(dom), tuplify(cod)
        packed_cls = channel.Channel[type(self).monad]
        self.inside = inside if isinstance(inside, packed_cls) else\
            packed_cls(
                lambda v: inside(*unpack_value(v)), pack(dom), pack(cod))
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom: Ty) -> Channel:
        """
        The identity channel on a tuple of wires ``dom``, given by the
        monad's unit.

        Parameters:
            dom : The tuple of wire types on which to take the identity.
        """
        dom = tuplify(dom)
        return cls(channel.Channel[cls.monad].id(pack(dom)), dom, dom)

    def raw(self, *xs):
        """ Call ``self``, keeping the packed monadic type as is. """
        return self.inside(pack_value(xs))

    def __call__(self, *xs):
        monad, raw = type(self).monad, self.raw(*xs)
        if len(self.cod) == 1:
            return raw
        unpacking = function.Function(unpack_value, pack(self.cod), tuple)
        return monad.functor(unpacking)(raw)

    def then(self, other: Channel) -> Channel:
        """
        The Kleisli composition of two channels, called with :code:`>>`.

        Parameters:
            other : The other channel to compose in sequence.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        return type(self)(
            self.inside >> other.inside, self.dom, other.cod)

    def right_whisker(self, Y: Ty) -> Channel:
        """
        ``self`` tensored with the identity on ``Y``, called with :code:`@`.

        Parameters:
            Y : The tuple of wire types to whisker on the right.
        """
        Y, monad, n = tuplify(Y), type(self).monad, len(self.dom)

        def inside(*xs):
            values, y = xs[:n], xs[n:]
            pairing = function.Function(
                lambda b: pack_value(unpack_value(b) + y),
                pack(self.cod), pack(self.cod + Y))
            return monad.functor(pairing)(self.raw(*values))
        return type(self)(inside, self.dom + Y, self.cod + Y)

    def left_whisker(self, X: Ty) -> Channel:
        """
        The identity on ``X`` tensored with ``self``, called with :code:`@`.

        Parameters:
            X : The tuple of wire types to whisker on the left.
        """
        X, monad, n = tuplify(X), type(self).monad, len(X)

        def inside(*xs):
            x, values = xs[:n], xs[n:]
            pairing = function.Function(
                lambda d: pack_value(x + unpack_value(d)),
                pack(self.cod), pack(X + self.cod))
            return monad.functor(pairing)(self.raw(*values))
        return type(self)(inside, X + self.dom, X + self.cod)

    def tensor(self, other: Channel) -> Channel:
        """
        The tensor of two channels, called with :code:`@` and biased so
        that the effects of ``self`` run before those of ``other``, see
        the module docstring.

        Parameters:
            other : The other channel to tensor in parallel.
        """
        assert_isinstance(other, type(self))
        return self.right_whisker(other.dom).then(
            other.left_whisker(self.cod))

    def __matmul__(self, other):
        return self.tensor(other) if isinstance(other, Channel)\
            else self.right_whisker(other)

    def __rmatmul__(self, other):
        return self.left_whisker(other)

    @classmethod
    def copy(cls, x: Ty, n: int = 2) -> Channel:
        """
        The channel making ``n`` copies of ``x``, given by the monad's
        unit applied to the ``n``-fold duplication of the input.

        Parameters:
            x : The tuple of wire types to copy.
            n : The number of copies.
        """
        x = tuplify(x)
        unit = cls.monad.unit(pack(n * x))
        return cls(lambda *xs: unit(pack_value(n * xs)), x, n * x)

    @classmethod
    def discard(cls, dom: Ty) -> Channel:
        """
        The channel discarding ``dom``, i.e. making zero copies.

        Parameters:
            dom : The tuple of wire types to discard.
        """
        return cls.copy(dom, 0)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"
