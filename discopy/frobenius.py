# -*- coding: utf-8 -*-

"""
The free hypergraph category, i.e. diagrams with swaps and spiders.

Note that spiders are also known as special commutative Frobenius algebras.
Diagrams in the free hypergraph category are faithfully as :class:`Hypergraph`,
see :cite:t:`BonchiEtAl22`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty
    Diagram
    Box
    Cup
    Cap
    Swap
    Spider
    Category
    Functor

Axioms
------

>>> H = Hypergraph

**Spiders**

We can check spider fusion, i.e. special commutative Frobenius algebra.

>>> x, y, z = map(Ty, "xyz")
>>> split, merge = H.spiders(1, 2, x), H.spiders(2, 1, x)
>>> unit, counit = H.spiders(0, 1, x), H.spiders(1, 0, x)

* (Co)commutative (co)monoid:

>>> assert unit @ x >> merge == H.id(x) == x @ unit >> merge
>>> assert merge @ x >> merge == x @ merge >> merge
>>> assert H.swap(x, x) >> merge == merge
>>> assert split >> counit @ x == H.id(x) == split >> x @ counit
>>> assert split >> split @ x == split >> x @ split
>>> assert split >> H.swap(x, x) == split

* Frobenius:

>>> assert split @ x >> x @ merge\\
...     == merge >> split\\
...     == x @ split >> merge @ x\\
...     == H.spiders(2, 2, x)

* Speciality:

>>> assert split >> merge == H.spiders(1, 1, x) == H.id(x)

* Coherence:

>>> assert H.spiders(0, 1, x @ x) == unit @ unit
>>> assert H.spiders(1, 0, x @ x) == counit @ counit
>>> assert H.spiders(1, 2, x @ x)\\
...     == split @ split >> x @ H.swap(x, x) @ x
>>> assert H.spiders(2, 1, x @ x)\\
...     == x @ H.swap(x, x) @ x >> merge @ merge

**Snakes**

Special commutative Frobenius algebras imply compact-closedness, i.e.

* Snake equations:

>>> left_snake = lambda x: H.id(x.r).transpose(left=True)
>>> right_snake = lambda x: H.id(x.l).transpose(left=False)
>>> assert left_snake(x) == H.id(x) == right_snake(x)
>>> assert left_snake(x @ y) == H.id(x @ y) == right_snake(x @ y)

* Yanking (a.k.a. Reidemeister move 1):

>>> right_loop = lambda x: x @ H.caps(x, x.r)\\
...     >> H.swap(x, x) @ x.r >> x @ H.cups(x, x.r)
>>> left_loop = lambda x: H.caps(x.r, x) @ x\\
...     >> x.r @ H.swap(x, x) >> H.cups(x.r, x) @ x
>>> top_loop = lambda x: H.caps(x, x.r) >> H.swap(x, x.r)
>>> bottom_loop = lambda x: H.swap(x, x.r) >> H.cups(x.r, x)
>>> reidemeister1 = lambda x:\\
...     top_loop(x) == H.caps(x.r, x) and bottom_loop(x) == H.cups(x, x.r)\\
...     and left_loop(x) == H.id(x) == right_loop(x)
>>> assert reidemeister1(x) and reidemeister1(x @ y) and reidemeister1(Ty())

* Coherence:

>>> assert H.caps(x @ y, y @ x)\\
...     == H.caps(x, x) @ H.caps(y, y) >> x @ H.swap(x, y @ y)\\
...     == H.spiders(0, 2, x @ y) >> x @ y @ H.swap(x, y)
>>> assert H.caps(x, x) >> H.cups(x, x) == H.spiders(0, 0, x)

**Swaps**

We can also check that the axioms for symmetry hold on the nose.

* Involution (a.k.a. Reidemeister move 2):

>>> reidermeister2 = lambda x, y: H.swap(x, y) >> H.swap(y, x) == H.id(x @ y)
>>> assert reidermeister2(x, y) and reidermeister2(x @ y, z)

* Yang-Baxter (a.k.a. Reidemeister move 3):

>>> left = H.swap(x, y) @ z\\
...     >> y @ H.swap(x, z)\\
...     >> H.swap(y, z) @ x
>>> right = x @ H.swap(y, z)\\
...     >> H.swap(x, z) @ y\\
...     >> z @ H.swap(x, y)
>>> assert left == right

* Coherence (a.k.a. pentagon equations):

>>> assert H.swap(x, y @ z) == H.swap(x, y) @ z >> y @ H.swap(x, z)
>>> assert H.swap(x @ y, z) == x @ H.swap(y, z) >> H.swap(x, z) @ y

* Naturality:

>>> f = H.box("f", x, y)
>>> assert f @ z >> H.swap(f.cod, z) == H.swap(f.dom, z) >> z @ f
>>> assert z @ f >> H.swap(z, f.cod) == H.swap(z, f.dom) >> f @ z
"""

from __future__ import annotations

from collections.abc import Callable

from discopy import compact, pivotal, hypergraph
from discopy.cat import factory
from discopy.monoidal import assert_isatomic
from discopy.utils import factory_name


class Ob(pivotal.Ob):
    """
    A frobenius object is a self-dual pivotal object.

    Parameters:
        name : The name of the object.
    """
    l = r = property(lambda self: self)


@factory
class Ty(pivotal.Ty):
    """
    A frobenius type is a pivotal type with frobenius objects inside.

    Parameters:
        inside (frobenius.Ob) : The objects inside the type.
    """
    ob_factory = Ob


@factory
class Diagram(compact.Diagram):
    """
    A frobenius diagram is a compact diagram with :class:`Spider` boxes.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ty_factory = Ty

    @classmethod
    def caps(cls, left, right):
        return cls.cups(left, right).dagger()

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Ty, phases=None
                ) -> Diagram:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out`` and
        some optional vector of ``phases``.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
            phases : The phase for each spider.
        """
        return interleaving(cls, cls.spider_factory)(
            n_legs_in, n_legs_out, typ, phases)

    def unfuse(self) -> Diagram:
        """
        Unfuse arbitrary spiders into spiders with one or three legs.

        See Also
        --------
        This calls :func:`coherence`.

        Example
        -------
        >>> from discopy.drawing import Equation
        >>> spider = Spider(3, 5, Ty(''), "$\\\\phi$") @ Ty()
        >>> Spider.color = "red"
        >>> Equation(spider, spider.unfuse(), symbol="$\\\\mapsto$").draw(
        ...     path='docs/_static/hypergraph/unfuse.png')

        .. image:: /_static/hypergraph/unfuse.png
            :align: center
        """
        F = compact.Functor(
            ob=lambda x: x, ar=lambda f:
                f.unfuse() if isinstance(f, Spider) else f)
        F.dom, F.cod = Category(), Category()
        return F(self)


class Box(compact.Box, Diagram):
    """
    A frobenius box is a compact box in a frobenius diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (compact.Box, )


class Cup(compact.Cup, Box):
    """
    A frobenius cup is a compact cup in a frobenius diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (compact.Cup, )


class Cap(compact.Cap, Box):
    """
    A frobenius cap is a compact cap in a frobenius diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (compact.Cap, )


class Swap(compact.Swap, Box):
    """
    A frobenius swap is a compact swap in a frobenius diagram.

    Parameters:
        left (Ty) : The type on the top left and bottom right.
        right (Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (compact.Swap, )


class Spider(Box):
    """
    The spider with :code:`n_legs_in` and :code:`n_legs_out`
    on a given atomic type, with some optional phase as ``data``.

    Parameters:
        n_legs_in : The number of legs in.
        n_legs_out : The number of legs out.
        typ : The type of the spider.
        data : The phase of the spider.

    Examples
    --------
    >>> x = Ty('x')
    >>> spider = Spider(1, 2, x)
    >>> assert spider.dom == x and spider.cod == x @ x
    """
    draw_as_spider = True
    color = "black"

    def __init__(self, n_legs_in: int, n_legs_out: int, typ: Ty, data=None,
                 **params):
        assert_isatomic(typ)
        self.typ = typ
        str_data = "" if data is None else f", {data}"
        name = type(self).__name__\
            + f"({n_legs_in}, {n_legs_out}, {typ}{str_data})"
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        Box.__init__(self, name, dom, cod, data=data, **params)
        self.drawing_name = "" if not data else str(data)

    @property
    def phase(self):
        """ The phase of the spider. """
        return self.data

    def __repr__(self):
        phase_repr = "" if self.phase is None \
            else f", phase={repr(self.phase)}"
        return factory_name(type(self)) + \
            f"({len(self.dom)}, {len(self.cod)}, {repr(self.typ)}{phase_repr})"

    def dagger(self):
        phase = None if self.phase is None else -self.phase
        return type(self)(len(self.cod), len(self.dom), self.typ, phase)

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.typ, self.phase)

    def unfuse(self) -> Diagram:
        return coherence(self.factory, type(self))(
            len(self.dom), len(self.cod), self.typ, self.phase)


class Category(compact.Category):
    """
    A hypergraph category is a compact category with a method :code:`spiders`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(compact.Functor):
    """
    A hypergraph functor is a compact functor that preserves spiders.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category()

    def __call__(self, other):
        if isinstance(other, Spider):
            return self.cod.ar.spiders(
                len(other.dom), len(other.cod), self(other.typ))
        return super().__call__(other)


class Hypergraph(hypergraph.Hypergraph):
    ty_factory = Ty
    box_factory = Box


def interleaving(cls: type, factory: Callable
                 ) -> Callable[[int, int, Ty], Diagram]:
    """
    Take a ``factory`` for spiders of atomic types and extend it recursively.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for spiders of atomic types, e.g. :class:`Spider`.
    """
    def method(n_legs_in, n_legs_out, typ, phases=None):
        phases = phases or len(typ) * [None]
        result = cls.id().tensor(*[
            factory(n_legs_in, n_legs_out, x, p) for x, p in zip(typ, phases)])
        for i, t in enumerate(typ):
            for j in range(n_legs_in - 1):
                result <<= result.dom[:i * j + i + j] @ cls.swap(
                    t, result.dom[i * j + i + j:i * n_legs_in + j]
                ) @ result.dom[i * n_legs_in + j + 1:]
            for j in range(n_legs_out - 1):
                result >>= result.cod[:i * j + i + j] @ cls.swap(
                    result.cod[i * j + i + j:i * n_legs_out + j], t
                ) @ result.cod[i * n_legs_out + j + 1:]
        return result

    return method


def coherence(cls: type, factory: Callable
              ) -> Callable[[int, int, Ty], Diagram]:
    """
    Take a ``factory`` for spiders with one or three legs of atomic types
    and extend it recursively to arbitrary spiders of atomic types.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for spiders of atomic types, e.g. :class:`Spider`.

    See Also
    --------
    This is called by :meth:`frobenius.Diagram.unfuse`.

    Note
    ----
    If the spider has a non-trivial phase then we also output a phase shifter.

    Example
    -------
    >>> print(Spider(2, 2, Ty('x'), 0.5).unfuse())
    Spider(2, 1, x) >> Spider(1, 1, x, 0.5) >> Spider(1, 2, x)
    """
    def method(a, b, x, phase=None):
        if phase is not None:  # Coherence for phase shifters.
            return method(a, 1, x)\
                >> factory(1, 1, x, phase)\
                >> method(1, b, x)
        if (a, b) in [(0, 1), (1, 0), (2, 1), (1, 2)]:
            return factory(a, b, x)
        if (a, b) == (1, 1):  # Speciality: one-to-one spiders are identity.
            return cls.id(x)
        if a < b:  # Cut the work in two.
            return method(b, a, x[::-1]).rotate()
        if b != 1:
            return method(a, 1, x) >> method(1, b, x)
        if a % 2:  # We can now assume a is odd and b == 1.
            return method(a - 1, 1, x) @ x >> factory(2, 1, x)
        # We can now assume a is even and b == 1.
        half_spiders = method(a // 2, 1, x)
        return half_spiders @ half_spiders >> factory(2, 1, x)

    return method


Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.braid_factory, Diagram.spider_factory = Swap, Spider
Id = Diagram.id
