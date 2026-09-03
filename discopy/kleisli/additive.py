# -*- coding: utf-8 -*-

"""
The Kleisli category of a monad with disjoint union as tensor.

A channel from a disjoint union ``dom`` to a disjoint union ``cod`` is a
Python function from a tagged value of ``dom`` to the monadic type
``M(pack(cod))``, where :func:`pack` turns a disjoint union of wire types
into the single type a :class:`~discopy.kleisli.monad.Monad` already knows
how to act on: the bare wire type itself if there is only one, else
:class:`Tagged`, i.e. a value together with the index of the summand it
comes from. This mirrors :mod:`discopy.python.additive`, where a value of a
disjoint union is a pair ``(obj, tag)``, with the pair replaced by a class
of its own for the same reason as in
:mod:`discopy.kleisli.multiplicative`: a payload inside ``M`` flows through
composition as a single value, and a genuine tuple would be mistaken for
several values to splat.

Disjoint union makes this a genuinely monoidal category, unlike the tuple
tensor of :mod:`discopy.kleisli.multiplicative`: only one side of ``f + g``
ever runs, so there is no bias in the order of effects and no commutativity
condition.

Trace
-----

The trace over the last ``n`` summands feeds the outputs tagged in the
traced part back into the corresponding inputs, i.e. the *execution
formula*. Once channels are monadic this is more than a ``while`` loop: a
single monadic value holds both the outcomes that leave the loop and those
that go around again, and ``(functor, unit, mult)`` cannot tell them apart.
The trace is therefore defined exactly when the monad supplies an
:attr:`~discopy.kleisli.monad.Monad.iterate` operator

.. math::
    (-)^\\dagger : (U \\to M(Y + U)) \\to (U \\to M(Y))

i.e. when it is an `Elgot monad
<https://ncatlab.org/nlab/show/Elgot+monad>`_, see Adámek, Milius & Velebil,
`Elgot theories: a new perspective of the equational properties of iteration
<https://arxiv.org/abs/1006.0918>`_ (2010). The maybe, powerset and
subdistribution monads all are; a monad without one raises rather than
returning a wrong answer.

Token machines
--------------

Dal Lago & Hoshino's `Geometry of Bayesian Programming
<https://arxiv.org/abs/1904.11324>`_ (2019) reads a probabilistic program
as a net and its execution as a token walking through it. A machine with
entry wires ``dom``, exit wires ``cod`` and ``n`` internal wires is a
channel over the subdistribution monad

.. math::
    \\text{net} : \\text{dom} + \\text{mem} \\to \\text{cod} + \\text{mem}

sending the token at one port to the subdistribution of ports it may move
to next, and the machine's whole behaviour is ``net.trace(n)``: the
execution formula walks the token until it leaves through an exit.

Take a coin drawn from a fair one and one biased ``4 / 5`` towards heads,
then flipped, with a machine that exits when it comes up heads and sends
the token back to the start when it comes up tails:

>>> from discopy.kleisli.monad import Subdistribution
>>> Machine, Hyp, Unit = Channel[Subdistribution], str, type(None)
>>> draw = Machine(lambda _: frozenset({
...     ("fair", .5), ("biased", .5)}), (Unit, ), (Hyp, ))
>>> flip = Machine(lambda h: frozenset({
...     (Tagged(h, 0), .5 if h == "fair" else .8),
...     (Tagged(None, 1), .5 if h == "fair" else .2)}), (Hyp, ), (Hyp, Unit))
>>> net = Machine.merge((Unit, ), 2) >> draw >> flip

One step of the machine leaves the token spread over the exit and the
internal wire. The mass that exits is the evidence, i.e. the probability
of observing heads, and the mass that loops is what a subdistribution is
allowed to lose:

>>> token = (draw >> flip)(None)
>>> round(sum(p for port, p in token if port.tag == 0), 3)
0.65

Tracing the internal wire feeds the tails back into the draw, i.e. it is
rejection sampling written as a feedback loop. The execution formula
resolves it exactly rather than sampling, and what comes out is the
posterior of Bayes' rule, ``.25 / .65`` and ``.4 / .65``:

>>> {h: round(p, 3) for h, p in sorted(net.trace()(None))}
{'biased': 0.615, 'fair': 0.385}

Losing mass and then recovering it is the whole point: conditioning is a
trace, so a Bayesian program is a token machine whose net says nothing
about probability beyond one step at a time.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Tagged
    Channel
    pack
    pack_value
    unpack_value
    injection
"""
from __future__ import annotations

from collections.abc import Callable

from discopy.abc import NamedGeneric, TracedCategory
from discopy.kleisli import channel
from discopy.kleisli.monad import Monad
from discopy.python import function
from discopy.python.additive import Ty
from discopy.utils import (
    assert_isinstance, assert_iscomposable, factory, factory_name, tuplify)


class Tagged:
    """
    A value together with the index of the summand it comes from, i.e. an
    element of a disjoint union, see the module docstring for why this
    needs to be its own class rather than a plain :class:`tuple`.

    Parameters:
        value : The value itself.
        tag : The index of the summand it belongs to.

    Example
    -------
    >>> assert Tagged("x", 1) == eval(repr(Tagged("x", 1)))
    >>> assert Tagged("x", 1) != ("x", 1) != Tagged("x", 0)
    """
    def __init__(self, value, tag: int):
        self.value, self.tag = value, tag

    def __eq__(self, other):
        return isinstance(other, Tagged)\
            and (self.value, self.tag) == (other.value, other.tag)

    def __hash__(self):
        return hash((self.value, self.tag))

    def __repr__(self):
        return f"Tagged({self.value!r}, {self.tag!r})"


def pack(ty: Ty) -> type:
    """
    The Python type standing for a disjoint union of wires ``ty``, i.e. the
    single type a :class:`~discopy.kleisli.monad.Monad` acts on.

    Parameters:
        ty : The disjoint union of wire types to pack.

    Example
    -------
    >>> assert pack(()) is Tagged
    >>> assert pack((int, )) is int
    >>> assert pack((int, str)) is Tagged
    """
    ty = tuplify(ty)
    return ty[0] if len(ty) == 1 else Tagged


def pack_value(value, tag: int, ty: Ty):
    """
    Turn a value on the ``tag``-th wire of ``ty`` into a value of
    :func:`pack` of ``ty``.

    Parameters:
        value : The value to pack.
        tag : The index of the summand it belongs to.
        ty : The disjoint union of wire types.

    Example
    -------
    >>> assert pack_value("x", 0, (str, )) == "x"
    >>> assert pack_value("x", 1, (int, str)) == Tagged("x", 1)
    """
    return value if len(tuplify(ty)) == 1 else Tagged(value, tag)


def unpack_value(value) -> tuple:
    """
    Turn a value of a packed type back into a pair of a value and its tag,
    inverse to :func:`pack_value`.

    Parameters:
        value : The value to unpack.

    Example
    -------
    >>> assert unpack_value(Tagged("x", 1)) == ("x", 1)
    >>> assert unpack_value("x") == ("x", 0)
    """
    return (value.value, value.tag) if isinstance(value, Tagged)\
        else (value, 0)


def injection(offset: int, source: Ty, target: Ty) -> function.Function:
    """
    The injection of a disjoint union ``source`` into a bigger one
    ``target``, i.e. the function shifting every tag by ``offset``.

    Parameters:
        offset : The index of ``source``'s first summand inside ``target``.
        source : The disjoint union to inject.
        target : The disjoint union to inject it into.

    Example
    -------
    >>> assert injection(1, (str, ), (int, str))("x") == Tagged("x", 1)
    >>> assert injection(0, (int, str), (int, ))(Tagged(42, 0)) == 42
    """
    def inside(value):
        value, tag = unpack_value(value)
        return pack_value(value, tag + offset, target)
    return function.Function(inside, pack(source), pack(target))


@factory
class Channel(TracedCategory, NamedGeneric['monad']):
    """
    A channel is a morphism in the Kleisli category of a monad ``M`` with
    disjoint union as tensor, i.e. a Python function from a tagged value of
    ``dom`` to the packed monadic type ``M(pack(cod))``.

    Parameters:
        inside : The underlying function, from ``pack(dom)`` to
            ``M(pack(cod))``. It takes a bare value when ``dom`` has a
            single summand, else a value and its tag.
        dom : The domain, i.e. a disjoint union of wire types.
        cod : The codomain, i.e. a disjoint union of wire types.

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
    >>> assert (half >> half)(8) == 2 and (half >> half)(2) is None

    The tensor is the disjoint union, i.e. only one side ever runs:

    >>> increment = Safe(lambda x: x + 1, int, int)
    >>> either = half @ increment
    >>> assert either(8) == Tagged(4, 0) and either(8, 1) == Tagged(9, 1)

    A channel can be nondeterministic about which summand it lands in:

    >>> Nondet = Channel[Powerset]
    >>> both = Nondet(lambda x: frozenset({
    ...     Tagged(x, 0), Tagged(-x, 1)}), (int, ), (int, int))
    >>> assert both(1) == frozenset({Tagged(1, 0), Tagged(-1, 1)})
    """
    ob = Ty
    monad: Monad = None

    def __init__(self, inside: Callable, dom: Ty, cod: Ty):
        dom, cod = tuplify(dom), tuplify(cod)
        packed_cls = channel.Channel[type(self).monad]

        def call(value):
            value, tag = unpack_value(value)
            return inside(value) if len(dom) == 1 else inside(value, tag)
        self.inside = inside if isinstance(inside, packed_cls) else\
            packed_cls(call, pack(dom), pack(cod))
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom: Ty) -> Channel:
        """
        The identity channel on a disjoint union ``dom``, given by the
        monad's unit.

        Parameters:
            dom : The disjoint union of wire types.
        """
        dom = tuplify(dom)
        return cls(channel.Channel[cls.monad].id(pack(dom)), dom, dom)

    def __call__(self, obj, tag: int = 0):
        return self.inside(pack_value(obj, tag, self.dom))

    def then(self, other: Channel) -> Channel:
        """
        The Kleisli composition of two channels, called with :code:`>>`.

        Parameters:
            other : The other channel to compose in sequence.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        return type(self)(self.inside >> other.inside, self.dom, other.cod)

    def tensor(self, other: Channel) -> Channel:
        """
        The disjoint union of two channels, called with :code:`@`: the
        input picks the side that runs, so unlike
        :meth:`discopy.kleisli.multiplicative.Channel.tensor` this is
        unbiased.

        Parameters:
            other : The other channel to tensor in parallel.
        """
        assert_isinstance(other, type(self))
        monad = type(self).monad
        dom, cod = self.dom + other.dom, self.cod + other.cod
        left = monad(injection(0, self.cod, cod))
        right = monad(injection(len(self.cod), other.cod, cod))

        def inside(obj, tag=0):
            return left(self(obj, tag)) if tag < len(self.dom)\
                else right(other(obj, tag - len(self.dom)))
        return type(self)(inside, dom, cod)

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Channel:
        """
        The channel swapping the tags of a disjoint union from ``x + y`` to
        ``y + x``, given by the monad's unit.

        Parameters:
            x : The disjoint union on the left.
            y : The disjoint union on the right.
        """
        x, y = tuplify(x), tuplify(y)
        unit = cls.monad.unit(pack(y + x))

        def inside(obj, tag=0):
            tag = tag + len(y) if tag < len(x) else tag - len(x)
            return unit(pack_value(obj, tag, y + x))
        return cls(inside, x + y, y + x)

    @classmethod
    def merge(cls, x: Ty, n: int = 2) -> Channel:
        """
        The channel merging ``n`` copies of ``x`` into one, i.e. the
        codiagonal of the coproduct, given by the monad's unit.

        Parameters:
            x : The disjoint union of wire types to merge.
            n : The number of copies to merge.
        """
        x = tuplify(x)
        unit = cls.monad.unit(pack(x))
        return cls(
            lambda obj, tag=0: unit(pack_value(obj, tag % len(x), x)),
            n * x, x)

    def trace(self, n: int = 1, left: bool = False) -> Channel:
        """
        The additive trace of a channel, i.e. the execution formula: the
        outputs tagged in the last ``n`` summands are fed back into the
        last ``n`` inputs until every outcome has left the loop.

        Parameters:
            n : The number of summands to trace over.
            left : Whether to trace the first rather than the last
                summands, not implemented.

        Raises:
            ValueError : If the monad has no
                :attr:`~discopy.kleisli.monad.Monad.iterate` operator, i.e.
                if it is not known to be an Elgot monad.

        Example
        -------
        Halving until the result is odd, i.e. taking out the powers of two:

        >>> from discopy.kleisli.monad import Maybe
        >>> Safe = Channel[Maybe]
        >>> f = Safe(lambda x, tag=0: Tagged(
        ...     x // 2, 1) if x % 2 == 0 else Tagged(x, 0), (int, int),
        ...     (int, int))
        >>> assert f.trace()(12) == 3 and f.trace()(7) == 7

        A loop that never exits gives the empty set of outcomes:

        >>> from discopy.kleisli.monad import Powerset
        >>> loop = Channel[Powerset](lambda x, tag=0: frozenset({
        ...     Tagged(x, 1)}), (int, int), (int, int))
        >>> assert loop.trace()(0) == frozenset()
        """
        if n == 0:
            return self
        if left:
            raise NotImplementedError
        monad = type(self).monad
        if monad.iterate is None:
            raise ValueError(
                f"The monad {monad} has no iteration operator, so the "
                f"Kleisli category of {monad} has no trace.")
        dom, cod = self.dom[:-n], self.cod[:-n]

        def step(value):
            value, tag = unpack_value(value)
            return self(value, tag - len(cod) + len(dom))
        resolve = monad.iterate(
            step, lambda value: unpack_value(value)[1] < len(cod))
        exit_ = monad(injection(0, self.cod, cod))
        return type(self)(
            lambda obj, tag=0: exit_(resolve(self(obj, tag))), dom, cod)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.inside!r}, dom={self.dom!r}, cod={self.cod!r})"
