# -*- coding: utf-8 -*-

"""
The category of Python functions with disjoint union as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Function
    Hypergraph
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from discopy.abc import SymmetricCategory
from discopy.utils import assert_isinstance, tuplify, AxiomError
from discopy.python import function


""" Lists of types interpreted as disjoint union. """
Ty = tuple[type, ...]


class Function(function.Function, SymmetricCategory):
    """
    Python functions with disjoint union as tensor.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its list of input types.
        cod : The codomain of the function, i.e. its list of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            trace
    """

    ob = Ty

    def __init__(self, inside, dom, cod, is_swap_of=None):
        self.is_swap_of = is_swap_of
        super().__init__(inside, dom, cod)

    def __call__(self, obj, tag=0):
        if self.type_checking:
            assert_isinstance(obj, self.dom[tag])
        result = self.inside(obj, *(() if len(self.dom) == 1 else (tag, )))
        if self.type_checking:
            obj, tag = (result, 0) if len(self.cod) == 1 else result
            assert_isinstance(obj, self.cod[tag])
        return result

    def tensor(self, other: Function) -> Function:
        """
        The disjoint union of two functions, called with :code:`@`.

        Parameters:
            other : The other function to compose in sequence.
        """
        dom, cod = self.dom + other.dom, self.cod + other.cod

        def inside(obj, tag=0):
            if tag < len(self.dom):
                result = self(obj, tag)
                obj, tag = (result, 0) if len(self.cod) == 1 else result
            else:
                result = other(obj, tag - len(self.dom))
                obj, tag = (result, 0) if len(other.cod) == 1 else result
                tag += len(self.cod)
            return obj if len(cod) == 1 else (obj, tag)
        return Function(inside, dom, cod)

    @staticmethod
    @cache
    def swap(x: Ty, y: Ty) -> Function:
        """
        Swap the tags of a disjoint union from `x + y` to `y + x`.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        x, y = map(tuplify, (x, y))

        def inside(obj, tag=0):
            new_tag = tag + len(y) if tag < len(x) else tag - len(x)
            if len(x + y) == 1:
                assert new_tag == 0
                return obj
            return (obj, new_tag)
        return Function(inside, dom=x + y, cod=y + x, is_swap_of=(x, y))

    def dagger(self):
        if self.is_swap_of is None:
            raise ValueError
        return Function.swap(*self.is_swap_of[::-1])

    def trace(self, n=1, left=False):
        """
        The additive trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        if left:
            raise NotImplementedError
        dom, cod = self.dom[:-n], self.cod[:-n]

        def inside(obj, tag=0):
            run_at_least_once = True
            while run_at_least_once or tag >= len(cod):
                run_at_least_once = False
                result = self(obj, tag)
                obj, tag = (result, 0) if len(self.cod) == 1 else result
            return obj if len(cod) == 1 else result
        return Function(inside, dom, cod)

    @staticmethod
    def merge(x: Ty, n=2) -> Function:
        def inside(obj, tag=0):
            if len(x) == 1:
                assert tag % len(x) == 0
                return obj
            return (obj, tag % len(x))
        return Function(inside, n * x, x)


Swap = Function.braid = Function.swap
Id = Function.twist = Function.id
Merge = Function.merge


class Hypergraph:
    """
    A hypergraph of additive (token-passing) Python functions, i.e. a string
    diagram of :class:`Function` in the category with disjoint union as tensor,
    drawn as a graph of boxes connected by spiders.

    Parameters:
        dom (Ty) : The domain, i.e. the tuple of input summands.
        cod (Ty) : The codomain, i.e. the tuple of output summands.
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

    Note
    ----
    Where :class:`discopy.python.multiplicative.Hypergraph` carries data along
    every wire at once (copy-discard, **left**-monogamous, causal), this one
    carries a single **token** that hops from wire to wire, as in the
    Geometry of Interaction. A wire therefore needs a unique *consumer* (the
    one place the token goes next) but may have many *producers* (the token
    can arrive on it from different places on different passes) -- the dual
    condition, **right-monogamy**, i.e. the hypergraph lives in a *cocartesian*
    (disjoint-union) category.

    Causality is **not** required: cycles are allowed, and a cycle is run as a
    while loop, exactly as :meth:`Function.trace` runs a trace. The constructor
    raises :class:`AxiomError` if the hypergraph is not right-monogamous.

    Example
    -------
    A single box ``f : (int, int) -> (int, int)`` with its second output wired
    back to its second input is the trace of ``f``, i.e. a while loop:

    >>> def collatz(obj, tag=0):
    ...     if tag == 0:  # the value just entered the loop
    ...         return obj, 1
    ...     return (1, 0) if obj == 1 else (
    ...         (3 * obj + 1, 1) if obj % 2 else (obj // 2, 1))
    >>> f = Function(collatz, (int, int), (int, int))
    >>> loop = Hypergraph(
    ...     dom=(int, ), cod=(int, ), boxes=(f, ),
    ...     wires=((0, ), (((0, 1), (2, 1)), ), (2, )))
    >>> assert loop(27) == f.trace()(27) == 1

    The token enters on wire ``0``, the box sends it back out on wire ``1``
    (consumed again by the box) until it finally exits on wire ``2``.
    """
    ob = Ty

    def __init__(self, dom, cod, boxes, wires, spider_types=None):
        assert_isinstance(dom, tuple)
        assert_isinstance(cod, tuple)
        for box in boxes:
            assert_isinstance(box, Function)
        dom_wires, box_wires, cod_wires = wires
        self.dom, self.cod, self.boxes = tuple(dom), tuple(cod), tuple(boxes)
        self.dom_wires, self.cod_wires = tuple(dom_wires), tuple(cod_wires)
        self.box_wires = tuple(
            (tuple(box_dom), tuple(box_cod)) for box_dom, box_cod in box_wires)

        if len(self.dom_wires) != len(self.dom)\
                or len(self.cod_wires) != len(self.cod)\
                or len(self.box_wires) != len(self.boxes) or any(
                    len(box_dom) != len(box.dom)
                    or len(box_cod) != len(box.cod)
                    for box, (box_dom, box_cod)
                    in zip(self.boxes, self.box_wires)):
            raise ValueError(
                "The wires do not match the domain, codomain and boxes.")

        if spider_types is not None and not isinstance(spider_types, Mapping):
            spider_types = dict(enumerate(spider_types))
        self.spider_types = spider_types

        if not self.is_right_monogamous:
            raise AxiomError("Hypergraph is not right-monogamous.")

    @property
    def wires(self) -> tuple:
        """ The ``(dom_wires, box_wires, cod_wires)`` triple of spiders. """
        return self.dom_wires, self.box_wires, self.cod_wires

    @property
    def spiders(self) -> set:
        """ The set of spiders appearing in the wiring of the hypergraph. """
        return set().union(
            self.dom_wires, self.cod_wires, self.spider_types or (),
            *(box_dom + box_cod for box_dom, box_cod in self.box_wires))

    @property
    def spider_wires(self) -> tuple[dict, dict]:
        """
        A ``(producers, consumers)`` pair of mappings from each spider to the
        ports that the token can *leave* from (the domain inputs and the
        outputs of boxes) and the ports that the token *goes to* (the inputs of
        boxes and the codomain outputs).
        """
        producers = {spider: set() for spider in self.spiders}
        consumers = {spider: set() for spider in self.spiders}
        roles = [(producers, self.dom_wires)] + [
            role for box_dom, box_cod in self.box_wires
            for role in [(consumers, box_dom), (producers, box_cod)]] + [
            (consumers, self.cod_wires)]
        port = 0
        for role, wires in roles:
            for spider in wires:
                role[spider].add(port)
                port += 1
        return producers, consumers

    @property
    def is_right_monogamous(self) -> bool:
        """
        Checks right-monogamy, i.e. that each non-scalar spider is consumed by
        exactly one port -- the unique destination of the token on that wire.
        In that case the hypergraph lives in a cocartesian (disjoint-union)
        category: a wire may be produced any number of times.
        """
        producers, consumers = self.spider_wires
        return all(
            len(consumers[spider]) == 1
            for spider in self.spiders
            if producers[spider] or consumers[spider])

    @property
    def _consumer(self) -> dict:
        """
        Map each spider to its unique consumer: ``("output", j)`` for the
        ``j``-th overall output, or ``("box", i, p)`` for the ``p``-th input
        port of box ``i``.
        """
        result = {}
        for i, (box_dom, _) in enumerate(self.box_wires):
            for p, spider in enumerate(box_dom):
                result[spider] = ("box", i, p)
        for j, spider in enumerate(self.cod_wires):
            result[spider] = ("output", j)
        return result

    def __call__(self, obj, tag=0, max_steps=1_000_000):
        """
        Evaluate the hypergraph by passing a token through it.

        The token ``(obj, tag)`` enters on input wire ``tag``; at each wire it
        hops to the unique consumer, running a box (which moves the token from
        one of its input ports to one of its output ports) until it reaches an
        output wire. Cycles are run as while loops.

        Parameters:
            obj : The value carried by the token.
            tag : The input wire on which the token enters.
            max_steps : A guard against non-terminating token trajectories.
        """
        if not 0 <= tag < len(self.dom_wires):
            raise ValueError(f"Invalid input wire: {tag}")
        consumer, spider = self._consumer, self.dom_wires[tag]
        for _ in range(max_steps):
            kind, i, *port = consumer[spider]
            if kind == "output":
                return obj if len(self.cod) == 1 else (obj, i)
            box = self.boxes[i]
            result = box(obj, *port)
            obj, out_port = (result, 0) if len(box.cod) == 1 else result
            spider = self.box_wires[i][1][out_port]
        raise RuntimeError(f"Token did not exit after {max_steps} steps.")
