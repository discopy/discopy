# -*- coding: utf-8 -*-

"""
Token machines for the lambda calculus, i.e. evaluation as token passing.

Dal Lago & Hoshino's `Geometry of Bayesian Programming
<https://arxiv.org/abs/1904.07425>`_ (LICS 2019, MSCS 31(6):633-681, 2021)
reads a probabilistic program as a proof structure and its execution as a
token travelling along its edges, the data it carries updated by a Mealy
machine at each node. Their calculus ``PCF_SS`` is PCF with a constant
``sample``, which draws a real number from the uniform distribution on the
unit interval, and a term constructor ``score``, whose evaluation
multiplies the weight of the current probabilistic branch, i.e. soft
conditioning. Their model is the category of measurable spaces and partial
measurable functions, of which the finitely supported
:attr:`~discopy.kleisli.monad.Subdistribution` monad is the discrete
analogue.

This module walks the term itself rather than its proof structure. A token
is a position in the term together with the environment it reads its
variables from and the stack of what is left to do, and it travels in two
directions: :class:`Down` into a subterm still to be evaluated, :class:`Up`
out of one carrying a :obj:`Value`. One transition is a channel in
:mod:`discopy.kleisli.additive`, from an entry plus the two directions to an
exit plus the two directions

.. math::
    \\text{step} : \\text{Down} + (\\text{Down} + \\text{Up})
    \\to \\text{Value} + (\\text{Down} + \\text{Up})

so that the machine is the trace of ``step`` over the two directions, i.e.
the execution formula walks the token until it leaves through the exit

.. math::
    \\text{step}^\\dagger : \\text{Down} \\to M(\\text{Value})

One transition knows nothing about the monad beyond the value it returns:
the looping is :meth:`~discopy.kleisli.additive.Channel.trace` and the
monad's :attr:`~discopy.kleisli.monad.Monad.iterate`, so the same machine
is probabilistic over :attr:`~discopy.kleisli.monad.Subdistribution`,
nondeterministic over :attr:`~discopy.kleisli.monad.Powerset` and partial
over :attr:`~discopy.kleisli.monad.Maybe`.

Bayesian programming
--------------------

A :class:`Machine` interprets the constants of a term: a callable is a
primitive waiting for its argument, anything else is the monadic value the
constant evaluates to. Over the subdistribution monad this is exactly
``PCF_SS``: ``sample`` is a constant whose value is drawn from a
distribution and ``score`` is a primitive returning a subdistribution whose
total mass is its argument, i.e. multiplying the weight of the branch it
runs in.

We take the discrete analogue of the Bayesian program those two constants
are there for: a coin of unknown bias has come up heads twice, what is its
bias? Draw the bias from a prior, score it by the likelihood of the data
and return it, the abstraction that ignores its argument being the
sequencing ``let _ = score(...) in p``.

>>> from discopy.closed import Ty
>>> from discopy.kleisli.monad import Subdistribution
>>> R, U = Ty("R"), Ty("U")
>>> sample, star = R("sample"), U("*")
>>> score, likelihood = (R >> U)("score"), (R >> R)("likelihood")
>>> infer = R(lambda p: U(lambda _: p)(score(likelihood(p))))
>>> print(infer)
R(lambda p: U(lambda _: p)((R >> U)('score')((R >> R)('likelihood')(p))))

The prior is uniform on three biases, the likelihood of two heads is the
square of the bias and scoring by a weight puts that much mass on the unit
value, i.e. it multiplies the weight of the branch it runs in:

>>> machine = Machine[Subdistribution]({
...     sample: frozenset({(p, 1 / 3) for p in (.25, .5, .75)}),
...     likelihood: lambda p: frozenset({(p * p, 1.)}),
...     score: lambda weight: frozenset({(star, weight)})})

The token comes out of the term with the unnormalised posterior, i.e. each
bias weighted by the probability of drawing it and then seeing two heads:

>>> bias = machine(infer(sample))
>>> {p: round(w, 3) for p, w in sorted(bias)}
{0.25: 0.021, 0.5: 0.083, 0.75: 0.188}

The mass the token lost on the way is what conditioning threw away, so what
is left is the probability of the data, i.e. the evidence, and dividing by
it is Bayes' rule:

>>> round(evidence(bias), 3)
0.292
>>> {p: round(w, 3) for p, w in sorted(posterior(bias))}
{0.25: 0.071, 0.5: 0.286, 0.75: 0.643}

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Closure
    Arg
    Fun
    Down
    Up
    Machine
    lookup
    evidence
    posterior
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from discopy.abc import NamedGeneric
from discopy.closed import (
    Abstraction, Application, Constant, TermBase, Variable)
from discopy.kleisli.additive import Channel, Tagged
from discopy.kleisli.monad import Monad
from discopy.python.function import Function
from discopy.utils import factory_name


Value = object
"""
What a token carries when it travels :class:`Up`, i.e. any Python object: a
:class:`Closure` for an abstraction, the constant itself for a constant the
:class:`Machine` does not interpret, else whatever payload its
interpretation gives, e.g. a float.
"""


@dataclass(frozen=True)
class Closure:
    """
    An abstraction together with the environment its free variables read
    from, i.e. the value of a function.

    Parameters:
        abstraction : The abstraction itself.
        env : The environment, see :func:`lookup`.
    """
    abstraction: TermBase
    env: tuple[tuple[Variable, Value], ...]


@dataclass(frozen=True)
class Arg:
    """
    The stack frame of an argument that is still a term, i.e. what is left
    to do once the function it is applied to has become a value.

    Parameters:
        term : The argument, still to be evaluated.
        env : The environment it is to be evaluated in.
    """
    term: TermBase
    env: tuple[tuple[Variable, Value], ...]


@dataclass(frozen=True)
class Fun:
    """
    The stack frame of a function that is already a value, i.e. what is left
    to do once its argument has become a value too.

    Parameters:
        value : The function, waiting for its argument.
    """
    value: Value


@dataclass(frozen=True)
class Down:
    """
    A token going down into a term still to be evaluated.

    Parameters:
        term : The position of the token, i.e. the subterm it sits on.
        env : The environment its free variables read from, see
            :func:`lookup`.
        stack : What is left to do once the term becomes a value, i.e. a
            tuple of :class:`Arg` and :class:`Fun` frames.
    """
    term: TermBase
    env: tuple[tuple[Variable, Value], ...] = ()
    stack: tuple[Arg | Fun, ...] = ()


@dataclass(frozen=True)
class Up:
    """
    A token coming up out of a term with the value it evaluated to.

    Parameters:
        value : The value the token carries.
        stack : What is left to do with it, see :class:`Down`.
    """
    value: Value
    stack: tuple[Arg | Fun, ...] = ()


directions = {Down: 1, Up: 2}
"""
The two directions a token travels in, together with the summand of
:attr:`Machine.channel` each of them belongs to, i.e. the memory that
:meth:`Machine.__call__` traces away.
"""


def lookup(env: tuple, variable: Variable) -> Value:
    """
    The value of a variable in an environment, i.e. a tuple of pairs with
    the innermost binder first, so that a variable shadows the ones it
    binds over.

    Parameters:
        env : The environment, innermost binder first.
        variable : The variable to look up.

    Raises:
        ValueError : If the variable is free in the environment.

    Example
    -------
    >>> from discopy.closed import Ty, Variable
    >>> x = Variable("x", Ty("X"))
    >>> assert lookup(((x, 42), (x, 0)), x) == 42
    >>> lookup((), x)
    Traceback (most recent call last):
    ...
    ValueError: The variable x is free, it has no value.
    """
    for binder, value in env:
        if binder == variable:
            return value
    raise ValueError(f"The variable {variable} is free, it has no value.")


def evidence(values: frozenset) -> float:
    """
    The mass a subdistribution has left, i.e. the probability of the data a
    Bayesian program conditioned on.

    Parameters:
        values : The subdistribution of values a :class:`Machine` returned.

    Example
    -------
    >>> assert evidence(frozenset({(0, .25), (1, .5)})) == .75
    """
    return sum(weight for _, weight in values)


def posterior(values: frozenset) -> frozenset:
    """
    A subdistribution divided by its :func:`evidence`, i.e. Bayes' rule.

    Parameters:
        values : The subdistribution of values a :class:`Machine` returned.

    Raises:
        ValueError : If the evidence is zero, i.e. if the program
            conditioned on something it never observed.

    Example
    -------
    >>> assert posterior(frozenset({(0, .25), (1, .25)}))\\
    ...     == frozenset({(0, .5), (1, .5)})
    >>> posterior(frozenset())
    Traceback (most recent call last):
    ...
    ValueError: The evidence is zero, there is no posterior.
    """
    total = evidence(values)
    if not total:
        raise ValueError("The evidence is zero, there is no posterior.")
    return frozenset((value, weight / total) for value, weight in values)


class Machine(NamedGeneric['monad']):
    """
    A token machine for the lambda calculus, i.e. an interpretation of the
    constants of a term, from which :attr:`channel` builds one transition
    and :meth:`__call__` traces it.

    Parameters:
        constants : What each constant of the term means, either a callable
            for a primitive waiting for its argument, or the monadic value
            a constant of base type evaluates to. The keys are the
            constants themselves rather than their names, so that two
            constants sharing a name but not their type do not alias. A
            constant with no interpretation is its own value, so applying
            one raises.

    Note
    ----
    The monad ``M`` is fixed by specialising the class with ``Machine[M]``
    and it must come with an
    :attr:`~discopy.kleisli.monad.Monad.iterate` operator, since the
    machine is a :meth:`~discopy.kleisli.additive.Channel.trace`.

    Example
    -------
    An abstraction evaluates to a closure and applying one substitutes,
    which takes no effect at all:

    >>> from discopy.closed import Ty
    >>> from discopy.kleisli.monad import Maybe
    >>> X = Ty("X")
    >>> identity, c = X(lambda x: x), X("c")
    >>> assert Machine[Maybe]()(identity) == Closure(identity, ())
    >>> assert Machine[Maybe]()(identity(c)) == c

    A nondeterministic machine is the same machine over the powerset monad,
    e.g. a coin whose two outcomes both come back:

    >>> from discopy.kleisli.monad import Powerset
    >>> B = Ty("B")
    >>> flip = B("flip")
    >>> machine = Machine[Powerset]({flip: frozenset({"heads", "tails"})})
    >>> assert machine(B(lambda x: x)(flip))\\
    ...     == frozenset({"heads", "tails"})
    """
    monad: Monad = None

    def __init__(self, constants: Mapping | None = None):
        self.constants = dict(constants or {})

    def inject(self, token: Down | Up):
        """
        A token tagged by the direction it travels in, then sent into the
        monad by its unit.

        Parameters:
            token : The token, going either down or up.
        """
        unit = type(self).monad.unit(Tagged)
        return unit(Tagged(token, directions[type(token)]))

    def leave(self, value: Value):
        """
        A value tagged by the exit, then sent into the monad by its unit,
        i.e. the token leaves the term with it.

        Parameters:
            value : The value the token carries out.

        Note
        ----
        The exit is a summand of its own rather than the absence of a
        direction, so that a value which happens to be a token, e.g. the
        interpretation of a constant, cannot re-enter the trace.
        """
        return type(self).monad.unit(Tagged)(Tagged(value, 0))

    def bind(self, values, resume: Callable):
        """
        Feed every value of a monadic value into ``resume``, i.e. the
        Kleisli composition of :mod:`discopy.kleisli.channel` written for a
        monadic value rather than a channel.

        Parameters:
            values : The monadic value to bind.
            resume : What to do with each of its values.
        """
        monad = type(self).monad
        lift = monad.functor(Function(resume, Value, monad(Tagged)))
        return monad.mult(Tagged)(lift(values))

    def descend(self, token: Down):
        """
        The transition of a token going down: into the function of an
        application, or back up with the closure of an abstraction, the
        value of a variable or the value of a constant.

        Parameters:
            token : The token going down.

        Raises:
            NotImplementedError : If the term is not a constant, a
                variable, an application or an abstraction.
        """
        term, env, stack = token.term, token.env, token.stack
        if isinstance(term, Application):
            return self.inject(
                Down(term.func, env, (Arg(term.args, env), ) + stack))
        if isinstance(term, Abstraction):
            return self.inject(Up(Closure(term, env), stack))
        if isinstance(term, Variable):
            return self.inject(Up(lookup(env, term), stack))
        if not isinstance(term, Constant):
            raise NotImplementedError(
                f"The token machine cannot walk through {term}.")
        effect = self.constants.get(term, term)
        if callable(effect):
            return self.inject(Up(term, stack))
        return self.bind(effect, lambda value: self.inject(Up(value, stack)))

    def ascend(self, token: Up):
        """
        The transition of a token going up, i.e. it leaves through the exit
        when nothing is left to do, goes down into an argument still to be
        evaluated, else applies the function it was waiting for.

        Parameters:
            token : The token going up.
        """
        value, stack = token.value, token.stack
        if not stack:
            return self.leave(value)
        frame, stack = stack[0], stack[1:]
        if isinstance(frame, Arg):
            return self.inject(
                Down(frame.term, frame.env, (Fun(value), ) + stack))
        return self.apply(frame.value, value, stack)

    def apply(self, func: Value, arg: Value, stack: tuple):
        """
        The transition applying a function value to an argument value, i.e.
        beta reduction for a closure and the interpretation of the constant
        for a primitive.

        Parameters:
            func : The function value.
            arg : The argument value.
            stack : What is left to do with the result.

        Raises:
            ValueError : If the function value is not a function, i.e. a
                closure or a constant with a callable interpretation.
        """
        if isinstance(func, Closure):
            env = ((func.abstraction.var, arg), ) + func.env
            return self.inject(Down(func.abstraction.body, env, stack))
        effect = self.constants.get(func)\
            if isinstance(func, Constant) else None
        if not callable(effect):
            raise ValueError(f"The value {func} is not a function.")
        return self.bind(
            effect(arg), lambda value: self.inject(Up(value, stack)))

    def step(self, state: Down | Up):
        """
        One transition of the machine, i.e. the token moves once.

        Parameters:
            state : The token, going either down or up.
        """
        return self.descend(state) if isinstance(state, Down)\
            else self.ascend(state)

    @property
    def channel(self) -> Channel:
        """
        One transition as a channel, from an entry plus the two directions
        of the token to an exit plus the two directions.
        """
        summands = tuple(directions)
        return Channel[type(self).monad](
            lambda state, tag=0: self.step(state),
            (Down, ) + summands, (Value, ) + summands)

    def __call__(self, term: TermBase):
        """
        The value of a term, i.e. the token enters at its root with an
        empty environment and an empty stack, and the trace of
        :attr:`channel` walks it until it exits.

        Parameters:
            term : The term to evaluate.
        """
        return self.channel.trace(len(directions))(Down(term))

    def __repr__(self):
        return factory_name(type(self)) + f"({self.constants!r})"
