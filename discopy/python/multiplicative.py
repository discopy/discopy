# -*- coding: utf-8 -*-

"""
The category of Python functions with tuple as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Function
    Hypergraph

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        exp
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from discopy.abc import ClosedCategory
from discopy.utils import (
    assert_isinstance, tuplify, untuplify, ar_factory, AxiomError)
from discopy.python import function


""" Functions have lists of types as input and output. """
Ty = tuple[type, ...]


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a tuple of Python types by another.

    Parameters:
        base (python.Ty) : The base type.
        exponent (python.Ty) : The exponent type.
    """
    return (Callable[list(exponent), tuple[base]], )


@ar_factory
class Function(function.Function, ClosedCategory):
    """
    Python function with tuple as tensor.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its list of input types.
        cod : The codomain of the function, i.e. its list of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            copy
            discard
            ev
            curry
            uncurry
            fix
            trace
    """

    ob = Ty

    def __call__(self, *xs):
        if self.type_checking:
            if len(xs) != len(self.dom):
                raise ValueError
            for (x, t) in zip(xs, self.dom):
                callable(x) or assert_isinstance(x, t)
        ys = self.inside(*xs)
        if self.type_checking:
            if len(self.cod) != 1 and (
                    not isinstance(ys, tuple) or len(self.cod) != len(ys)):
                raise RuntimeError
            for (y, t) in zip(tuplify(ys), self.cod):
                callable(y) or assert_isinstance(y, t)
        return ys

    def tensor(self, other: Function) -> Function:
        """
        The parallel composition of two functions, called with :code:`@`.

        Parameters:
            other : The other function to compose in sequence.
        """
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: Ty, y: Ty) -> Function:
        """
        The function for swapping two tuples of types :code:`x` and :code:`y`.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
        return Function(inside, dom=x + y, cod=y + x)

    braid = swap

    @staticmethod
    def copy(x: Ty, n=2) -> Function:
        """
        The function for making :code:`n` copies of a tuple of types :code:`x`.

        Parameters:
            x : The tuple of types to copy.
            n : The number of copies.
        """
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)

    @staticmethod
    def discard(dom: Ty) -> Function:
        """
        The function discarding a tuple of types, i.e. making zero copies.

        Parameters:
            dom : The tuple of types to discard.
        """
        return Function.copy(dom, 0)

    @staticmethod
    def ev(base: Ty, exponent: Ty, left=True) -> Function:
        """
        The evaluation function,
        i.e. take a function and apply it to an argument.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the function on the left or right.
        """
        if left:
            dom, cod = Function.exp(base, exponent) + exponent, base
            return Function(lambda f, *xs: f(*xs), dom, cod)
        dom, cod = exponent + Function.exp(base, exponent), base
        return Function(lambda *xs: xs[-1](*xs[:-1]), dom, cod)

    def curry(self, n=1, left=True) -> Function:
        """
        Currying, i.e. turn a binary function into a function-valued function.

        Parameters:
            n : The number of types to curry.
            left : Whether to curry on the left or right.
        """
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = Function.exp(self.cod, self.dom[len(self.dom) - n:])
        else:
            dom, cod = self.dom[n:], Function.exp(self.cod, self.dom[:n])
        return Function(dom=dom, cod=cod, inside=lambda *xs: lambda *ys:
                        self(*(xs + ys) if left else (ys + xs)))

    def uncurry(self, left=True) -> Function:
        """
        Uncurrying,
        i.e. turn a function-valued function into a binary function.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        traced = self.cod[0].__args__
        base, exponent = traced[-1].__args__, traced[:-1]
        return self @ exponent >> Function.ev(base, exponent) if left\
            else exponent @ self >> Function.ev(base, exponent, left=False)

    def fix(self, n=1) -> Function:
        """
        The parameterised fixed point of a function.

        Parameters:
            n : The number of types to take the fixed point over.
        """
        def inside(*xs, y=None):
            result = self.inside(*xs + (() if y is None else (y, )))
            return y if result == y else inside(*xs, y=result)
        return self if n == 0\
            else Function(inside, self.dom[:-1], self.cod).fix(n - 1)

    def trace(self, n=1, left=False):
        """
        The multiplicative trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        if left:
            raise NotImplementedError
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix()
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)

    exp = over = under = staticmethod(lambda x, y: exp(x, y))


class Hypergraph:
    """
    A hypergraph of Python functions, i.e. a string diagram of
    :class:`Function` drawn as a graph of boxes connected by spiders.

    Parameters:
        dom (Ty) : The domain, i.e. the tuple of input types.
        cod (Ty) : The codomain, i.e. the tuple of output types.
        boxes (tuple[Function, ...]) : The functions inside the hypergraph.
        wires : A :code:`(dom_wires, box_wires, cod_wires)` triple of spiders.
        spider_types (Mapping | Iterable) :
            Optional types for the spiders, indexed by spider label.

    The wiring is given as in :class:`discopy.hypergraph.Hypergraph`:

    - ``dom_wires`` is one spider for each input,
    - ``cod_wires`` is one spider for each output,
    - ``box_wires`` is a ``(box_dom_wires, box_cod_wires)`` pair of spiders for
      each box, listing the spider read by each of its input ports and the
      spider written by each of its output ports.

    A spider is any hashable label that identifies a wire; ports connected to
    the same spider carry the same value.

    Note
    ----
    The constructor checks that the hypergraph is **left-monogamous** and
    **causal**, i.e. that it lives in a copy-discard (a.k.a. cartesian or
    Markov) category: every wire is produced exactly once -- by an input or
    the output of a box -- but it may then be read zero times (discarded) or
    any number of times (copied), and the wires point forward so that the
    boxes can be evaluated in order. An :class:`AxiomError` is raised
    otherwise.

    Example
    -------
    >>> swap = Function(lambda x, y: (y, x), (int, int), (int, int))
    >>> add = Function(lambda x, y: x + y, (int, int), (int, ))

    The hypergraph below copies its first input into both addends, so that
    ``f(x, y) = (y + x, x)`` -- copying and discarding for free:

    >>> f = Hypergraph(
    ...     dom=(int, int), cod=(int, int),
    ...     boxes=(add, ),
    ...     wires=((0, 1), (((1, 0), (2, )), ), (2, 0)))
    >>> assert f(2, 3) == (5, 2)

    A box that reaches backwards is not causal:

    >>> Hypergraph((int, ), (int, ),
    ...     (swap, ), ((0, ), (((0, 1), (1, 2)), ), (2, )))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: Hypergraph is not causal.

    A wire produced twice is not left-monogamous:

    >>> Hypergraph((int, ), (int, ),
    ...     (add, ), ((0, ), (((0, 0), (0, )), ), (0, )))
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: Hypergraph is not left-monogamous.
    """
    category = Function
    ob = Ty

    def __init__(
            self, dom: Ty, cod: Ty, boxes: tuple[Function, ...],
            wires, spider_types=None):
        assert_isinstance(dom, tuple)
        assert_isinstance(cod, tuple)
        for box in boxes:
            assert_isinstance(box, self.category)
        self.dom, self.cod, self.boxes = tuple(dom), tuple(cod), tuple(boxes)
        dom_wires, box_wires, cod_wires = wires

        if len(dom_wires) != len(self.dom):
            raise ValueError
        if len(cod_wires) != len(self.cod):
            raise ValueError
        if len(box_wires) != len(self.boxes):
            raise ValueError
        for box, (box_dom_wires, box_cod_wires) in zip(self.boxes, box_wires):
            if len(box_dom_wires) != len(box.dom):
                raise ValueError
            if len(box_cod_wires) != len(box.cod):
                raise ValueError

        self.dom_wires = tuple(dom_wires)
        self.box_wires = tuple(
            (tuple(box_dom), tuple(box_cod)) for box_dom, box_cod in box_wires)
        self.cod_wires = tuple(cod_wires)
        self.wires = (self.dom_wires, self.box_wires, self.cod_wires)

        if spider_types is not None and not isinstance(spider_types, Mapping):
            spider_types = dict(enumerate(spider_types))
        self.spider_types = spider_types

        if not self.is_left_monogamous:
            raise AxiomError("Hypergraph is not left-monogamous.")
        if not self.is_causal:
            raise AxiomError("Hypergraph is not causal.")

    @property
    def spiders(self) -> set:
        """ The set of spiders appearing in the wiring of the hypergraph. """
        result = set(self.dom_wires) | set(self.cod_wires)
        for box_dom, box_cod in self.box_wires:
            result.update(box_dom)
            result.update(box_cod)
        if self.spider_types is not None:
            result.update(self.spider_types)
        return result

    @property
    def spider_wires(self) -> tuple[dict, dict]:
        """
        A ``(producers, consumers)`` pair of mappings from each spider to the
        set of port indices that write to it (its domain inputs and the
        outputs of boxes) and that read from it (the inputs of boxes and its
        codomain outputs), with ports indexed in evaluation order.
        """
        producers = {spider: set() for spider in self.spiders}
        consumers = {spider: set() for spider in self.spiders}
        port = 0
        for spider in self.dom_wires:
            producers[spider].add(port)
            port += 1
        for box_dom, box_cod in self.box_wires:
            for spider in box_dom:
                consumers[spider].add(port)
                port += 1
            for spider in box_cod:
                producers[spider].add(port)
                port += 1
        for spider in self.cod_wires:
            consumers[spider].add(port)
            port += 1
        return producers, consumers

    @property
    def is_left_monogamous(self) -> bool:
        """
        Checks left-monogamy, i.e. that each non-scalar spider is produced by
        exactly one port. In that case the hypergraph lives in a copy-discard
        category: a wire may then be read any number of times.
        """
        producers, consumers = self.spider_wires
        return all(
            len(producers[spider]) == 1
            for spider in self.spiders
            if producers[spider] or consumers[spider])

    @property
    def is_causal(self) -> bool:
        """
        Checks causality, i.e. that each non-scalar spider is produced by
        exactly one port and read only by ports that come after it. The boxes
        can then be evaluated in order, see :meth:`__call__`.
        """
        producers, consumers = self.spider_wires
        return all(
            len(producers[spider]) == 1 and all(
                producer < consumer
                for producer in producers[spider]
                for consumer in consumers[spider])
            for spider in self.spiders
            if producers[spider] or consumers[spider])

    def __call__(self, *xs):
        """
        Evaluate the hypergraph by dispatching to the functions inside.

        Takes one argument per input wire, feeds each box the values on its
        input wires (copying and discarding as the wiring dictates), and
        returns the tuple of values on the output wires -- or that value
        itself if there is a single output.

        Parameters:
            xs : One argument for each wire in :attr:`dom_wires`.
        """
        if len(xs) != len(self.dom_wires):
            raise ValueError
        values = dict(zip(self.dom_wires, xs))
        for box, (box_dom, box_cod) in zip(self.boxes, self.box_wires):
            ys = tuplify(box(*(values[spider] for spider in box_dom)))
            values.update(zip(box_cod, ys))
        return untuplify(tuple(values[spider] for spider in self.cod_wires))

    @classmethod
    def from_hypergraph(cls, source, ob, ar) -> Hypergraph:
        """
        Build a :class:`Hypergraph` of Python functions from a hypergraph in
        any other category, by relabeling its objects and boxes -- i.e. by
        applying the ``(ob, ar)`` data of a functor while keeping the wiring.

        Parameters:
            source (discopy.hypergraph.Hypergraph) : The hypergraph to map.
            ob (Mapping) : From each atomic type of ``source`` to a Python type.
            ar (Mapping) : From each box of ``source`` to a :class:`Function`.
        """
        translate = lambda typ: tuple(ob[atom] for atom in typ)
        spider_types = None if source.spider_types is None else tuple(
            ob[typ] for typ in source.spider_types)
        return cls(
            dom=translate(source.dom), cod=translate(source.cod),
            boxes=tuple(ar[box] for box in source.boxes),
            wires=source.wires, spider_types=spider_types)
