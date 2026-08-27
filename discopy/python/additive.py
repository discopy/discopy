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
"""

from __future__ import annotations

from functools import cache
from typing import Self

from discopy.abc import SymmetricCategory
from discopy.testing import Strategy
from discopy.utils import assert_isinstance, factory, tuplify
from discopy.python import finset, function
from discopy.python.function import Types


""" Lists of types interpreted as disjoint union. """
Ty = tuple[type, ...]


@factory
class Function(function.Function, SymmetricCategory, Strategy["Function"]):
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

    ob = Types

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

    @classmethod
    def permutation(cls, xs, doms) -> Self:
        """ Permute the tags of a disjoint union. """
        doms, xs = list(doms), finset.Permutation(xs, len(doms))
        offsets = [0]
        for dom in doms:
            offsets.append(offsets[-1] + len(dom))
        inverse = xs.dagger()

        def inside(obj, tag=0):
            block = next(i for i in range(len(doms))
                         if tag < offsets[i + 1])
            new_tag = sum(len(doms[i]) for i in xs[:inverse[block]])\
                + tag - offsets[block]
            return obj if offsets[-1] == 1 else (obj, new_tag)

        dom = sum(doms, ())
        cod = sum((doms[i] for i in xs), ())
        return cls(inside, dom, cod)

    def dagger(self):
        if self.is_swap_of is None:
            raise ValueError
        return Function.swap(*self.is_swap_of[::-1])

    dagger_involution = SymmetricCategory.dagger_involution.inapplicable(
        "Only a swap has a dagger.")

    dagger_contravariance = SymmetricCategory.dagger_contravariance\
        .inapplicable("Only a swap has a dagger.")

    dagger_monoidality = SymmetricCategory.dagger_monoidality.inapplicable(
        "Only a swap has a dagger.")

    @classmethod
    def equation_factory(cls, *terms):
        """
        Functions are compared extensionally, i.e. up to probing both
        sides on a canonical element of every tag.
        """
        from discopy.cat import Equation

        return Equation(*terms, up_to=cls.probe)

    @classmethod
    def probe(cls, f) -> tuple:
        """ The observations of a function on a canonical tagged element. """
        return tuple(
            f(seed, tag)
            for tag in range(len(f.dom)) for seed in (2, 3))

    @classmethod
    def strategy(cls, *, dom=None, cod=None, max_length=3, **_):
        """Generate tag relabellings, i.e. finite maps between the tags."""
        from hypothesis import strategies as st

        types = cls.ob.strategy(max_length=max_length)

        def functions(boundaries):
            source, target = map(tuplify, boundaries)
            if source and not target:
                return st.nothing()

            def build(mapping):
                def inside(obj, tag=0):
                    return obj if len(target) == 1 else (obj, mapping[tag])
                return cls(inside, source, target)
            return st.tuples(*(
                st.integers(min_value=0, max_value=len(target) - 1)
                for _ in source)).map(build)

        return st.tuples(
            types if dom is None else st.just(dom),
            types if cod is None else st.just(cod)).flatmap(functions)

    def trace(self, n=1, left=False):
        """
        The additive trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        if n == 0:
            return self
        if left:
            raise NotImplementedError
        dom, cod = self.dom[:-n], self.cod[:-n]

        def inside(obj, tag=0):
            run_at_least_once = True
            while run_at_least_once or tag >= len(cod):
                if not run_at_least_once:
                    tag = tag - len(cod) + len(dom)
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
