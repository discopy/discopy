# -*- coding: utf-8 -*-

"""
The ribbon category of representations of a finite-dimensional Hopf algebra.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    HopfAlgebra
    Representation
    Intertwiner
    Functor

A finite-dimensional quasitriangular Hopf algebra :math:`H` has a category of
representations :math:`\\mathrm{Rep}(H)` which is a ribbon category: the
braiding is the universal R-matrix, cups and caps come from the antipode, and
the twist is the trace of the braid. That category is :class:`Intertwiner` — a
category of :class:`.tensor.Diagram`\\ s whose objects are
:class:`Representation`\\ s, so its ribbon structure lives in its classmethods
:meth:`Intertwiner.braid`, :meth:`Intertwiner.twist`, ``cups`` and ``caps``. A
quantum topological invariant of tangles is then a ribbon :class:`Functor` from
the free :mod:`.ribbon` category into :class:`Intertwiner`, evaluated as
concrete tensors (see :mod:`.tensor`).

The structural generators of a :class:`HopfAlgebra` (and of a
:class:`Representation`) are stored as :class:`.tensor.Diagram`\\ s, so that
composing them, e.g. ``H.comult >> H.mult``, builds a fine-grained network; the
network is only contracted (a single ``einsum``) when a morphism is evaluated —
by the axiom checks, or by the ribbon :class:`Functor` on a knot.

Example
-------
The Drinfeld double of the group algebra of :math:`\\mathbb{Z}/2` gives a
non-trivial link invariant: it separates the Hopf link from the two-component
unlink.

>>> import numpy as np
>>> from discopy import ribbon
>>> H = HopfAlgebra.cyclic(2).double()
>>> assert H.is_valid() and H.dim == 4
>>> e = Representation.anyon(H, 0, -1)
>>> m = Representation.anyon(H, 1, 1)
>>> V = Representation.direct_sum([e, m])
>>> assert V.is_module()
>>> x = ribbon.Ty('x')
>>> F = Functor(ob_map={x: V}, ar_map={})
>>> braid = ribbon.Braid(x, x)
>>> hopf_link = (braid >> braid).trace(n=2)
>>> unlink = (ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)) @ (
...     ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r))

The functor gives the tensor network of each knot; contract it with ``.eval``:

>>> hopf = complex(F(hopf_link).eval(dtype=complex))
>>> split = complex(F(unlink).eval(dtype=complex))
>>> assert not np.isclose(hopf, split)
>>> assert np.isclose(hopf, 0) and np.isclose(split, 4)

Axioms
------
Because the generators are diagrams, an axiom is an equation of diagrams,
checked by contracting both sides (e.g. :meth:`HopfAlgebra.is_associative`).
We take a concrete example, the group algebra :math:`k[\\mathbb{Z}/2]`
(generators :math:`\\nabla`, :math:`\\Delta`, ...), and both draw *and* assert
each axiom.

>>> from discopy.drawing import Equation
>>> from discopy.tensor import Diagram
>>> H = HopfAlgebra.cyclic(2)
>>> ty = H.ty
>>> assert H.is_valid()

Associativity, ``(nabla @ H) >> nabla == (H @ nabla) >> nabla``:

>>> lhs, rhs = H.mult @ ty >> H.mult, ty @ H.mult >> H.mult
>>> assert H.is_associative()
>>> Equation(lhs, rhs).draw(path='docs/_static/hopf/associativity.png')

.. image:: /_static/hopf/associativity.png
    :align: center

The bialgebra law, ``Delta`` is an algebra homomorphism:

>>> lhs = H.mult >> H.comult
>>> rhs = H.comult @ H.comult >> ty @ Diagram.swap(ty, ty) @ ty \\
...     >> H.mult @ H.mult
>>> assert H.is_bialgebra()
>>> Equation(lhs, rhs).draw(path='docs/_static/hopf/bialgebra.png')

.. image:: /_static/hopf/bialgebra.png
    :align: center

The antipode axiom, ``comult >> (S @ H) >> mult == counit >> unit``:

>>> left = H.comult >> H.antipode @ ty >> H.mult
>>> right = H.comult >> ty @ H.antipode >> H.mult
>>> unit = H.counit >> H.unit
>>> assert H.has_antipode()
>>> Equation(left, unit, right).draw(path='docs/_static/hopf/antipode.png')

.. image:: /_static/hopf/antipode.png
    :align: center

The derived generators of the double are *composites* of the base generators
and cups/caps — never materialised. Here is its multiplication, the coadjoint
product of :math:`D(H) = H \\otimes H^*`, drawn as a :class:`.tensor.CMap` (the
tensor network that gets contracted):

>>> HopfAlgebra.cyclic(2).double().mult.to_map().draw(
...     path='docs/_static/hopf/double_mult.png')

.. image:: /_static/hopf/double_mult.png
    :align: center
"""

from __future__ import annotations

import numpy as np

from discopy import ribbon, tensor, frobenius
from discopy.tensor import Dim, Box, Id
from discopy.abc import RibbonCategory
from discopy.utils import factory_name, product

Diagram = tensor.Diagram


class HopfAlgebra:
    """
    A finite-dimensional Hopf algebra whose structural generators are
    :class:`.tensor.Diagram`\\ s over one object ``ty``:

    ==============  ==============================
    ``unit``        :math:`1 \\to H` (:math:`\\eta`)
    ``counit``      :math:`H \\to 1` (:math:`\\epsilon`)
    ``mult``        :math:`H \\otimes H \\to H` (:math:`\\nabla`)
    ``comult``      :math:`H \\to H \\otimes H` (:math:`\\Delta`)
    ``antipode``    :math:`H \\to H` (:math:`S`)
    ``R``           :math:`1 \\to H \\otimes H` (optional, the R-matrix)
    ==============  ==============================

    An axiom is a diagram equation, checked by evaluating both sides (see
    :meth:`is_valid`). Build one from concrete structure arrays with
    :meth:`from_arrays`, or compose generators to derive new algebras (see
    :meth:`double`).

    Parameters:
        unit, counit, mult, comult, antipode : The generators as diagrams.
        R : The R-matrix generator (optional).
        antipode_inv : The inverse antipode (optional, used by :meth:`double`).
    """
    def __init__(self, unit, counit, mult, comult, antipode,
                 R=None, antipode_inv=None):
        self.unit, self.counit = unit, counit
        self.mult, self.comult, self.antipode = mult, comult, antipode
        self.R, self.antipode_inv = R, antipode_inv
        self.ty = mult.cod
        self.dim = product(self.ty.inside)

    @property
    def generators(self):
        """ The tuple of structural generators, in constructor order. """
        return (self.unit, self.counit, self.mult, self.comult,
                self.antipode, self.R, self.antipode_inv)

    def __repr__(self):
        optional = "".join(
            f", {name}={value!r}" for name, value in
            [("R", self.R), ("antipode_inv", self.antipode_inv)]
            if value is not None)
        return factory_name(type(self)) + (
            f"(unit={self.unit!r}, counit={self.counit!r}, "
            f"mult={self.mult!r}, comult={self.comult!r}, "
            f"antipode={self.antipode!r}{optional})")

    def __eq__(self, other):
        return isinstance(other, HopfAlgebra) \
            and self.generators == other.generators

    @classmethod
    def from_arrays(cls, unit, counit, mult, comult, antipode, R=None):
        """
        Build a Hopf algebra from its structure arrays with respect to a basis
        :math:`e_0, \\dots, e_{n-1}`: ``mult[i, j, k]`` is the coefficient of
        :math:`e_k` in :math:`e_i e_j`, ``comult[i, p, q]`` of
        :math:`e_p \\otimes e_q` in :math:`\\Delta(e_i)`, ``antipode[i, j]`` of
        :math:`e_j` in :math:`S(e_i)`, and ``R[i, j]`` of :math:`e_i \\otimes
        e_j` in the R-matrix. Each is wrapped into a named
        :class:`.tensor.Box`.
        """
        ty = Dim(len(np.asarray(unit)))
        antipode = np.asarray(antipode, dtype=complex)

        def gen(name, dom, cod, data):
            array = np.asarray(data, dtype=complex).reshape(-1)
            return Box[complex](name, dom, cod, array.tolist())

        return cls(
            unit=gen('η', Dim(1), ty, unit),
            counit=gen('ε', ty, Dim(1), counit),
            mult=gen('∇', ty @ ty, ty, mult),
            comult=gen('Δ', ty, ty @ ty, comult),
            antipode=gen('S', ty, ty, antipode),
            R=None if R is None else gen('R', Dim(1), ty @ ty, R),
            antipode_inv=gen('S⁻¹', ty, ty, np.linalg.inv(antipode)))

    def is_associative(self):
        """ ``(mult @ ty) >> mult == (ty @ mult) >> mult``. """
        ty = self.ty
        return (self.mult @ ty >> self.mult).eval(dtype=complex).is_close(
            (ty @ self.mult >> self.mult).eval(dtype=complex))

    def is_unital(self):
        """ The unit is a left and right identity for ``mult``. """
        ty = self.ty
        return all(
            lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
            for lhs, rhs in [
                (self.unit @ ty >> self.mult, Id(ty)),
                (ty @ self.unit >> self.mult, Id(ty))])

    def is_coassociative(self):
        """ ``comult >> (comult @ ty) == comult >> (ty @ comult)``. """
        ty = self.ty
        return (self.comult >> self.comult @ ty).eval(dtype=complex).is_close(
            (self.comult >> ty @ self.comult).eval(dtype=complex))

    def is_counital(self):
        """ The counit is a left and right identity for ``comult``. """
        ty = self.ty
        return all(
            lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
            for lhs, rhs in [
                (self.comult >> self.counit @ ty, Id(ty)),
                (self.comult >> ty @ self.counit, Id(ty))])

    def is_commutative(self):
        """ ``Swap >> mult == mult`` (a *property*, not an axiom). """
        ty = self.ty
        return (Diagram.swap(ty, ty) >> self.mult).eval(
            dtype=complex).is_close(self.mult.eval(dtype=complex))

    def is_cocommutative(self):
        """ ``comult >> Swap == comult`` (a *property*, not an axiom). """
        ty = self.ty
        return (self.comult >> Diagram.swap(ty, ty)).eval(
            dtype=complex).is_close(self.comult.eval(dtype=complex))

    def is_bialgebra(self):
        """ ``comult`` and ``counit`` are algebra homomorphisms. """
        ty = self.ty
        return all(
            lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
            for lhs, rhs in [
                (self.mult >> self.comult,
                 self.comult @ self.comult >> ty @ Diagram.swap(ty, ty) @ ty
                 >> self.mult @ self.mult),
                (self.mult >> self.counit, self.counit @ self.counit),
                (self.unit >> self.comult, self.unit @ self.unit),
                (self.unit >> self.counit, Id(Dim(1)))])

    def has_antipode(self):
        """ ``comult >> (S @ ty) >> mult == counit >> unit ==
        comult >> (ty @ S) >> mult``. """
        ty, unit = self.ty, self.counit >> self.unit
        return all(
            lhs.eval(dtype=complex).is_close(unit.eval(dtype=complex))
            for lhs in [
                self.comult >> self.antipode @ ty >> self.mult,
                self.comult >> ty @ self.antipode >> self.mult])

    def is_quasitriangular(self):
        """
        Whether ``R`` is a universal R-matrix: it intertwines ``comult`` with
        its opposite, :math:`R \\Delta = \\Delta^{op} R`, and satisfies the two
        hexagon equations :math:`(\\Delta \\otimes 1) R = R_{13} R_{23}` and
        :math:`(1 \\otimes \\Delta) R = R_{13} R_{12}`.
        """
        if self.R is None:
            return False
        ty, R, m, d = self.ty, self.R, self.mult, self.comult
        swap = Diagram.swap(ty, ty)
        return all(
            lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
            for lhs, rhs in [
                (R @ d >> ty @ swap @ ty >> m @ m,
                 (d >> swap) @ R >> ty @ swap @ ty >> m @ m),
                (R >> d @ ty,
                 R @ R >> ty @ swap @ ty >> ty @ ty @ m),
                (R >> ty @ d,
                 R @ R >> ty @ swap @ ty >> m @ ty @ ty >> ty @ swap)])

    def is_valid(self):
        """
        Whether the axioms of a Hopf algebra hold, and quasitriangularity
        whenever there is an R-matrix.
        """
        return self.is_associative() and self.is_unital() \
            and self.is_coassociative() and self.is_counital() \
            and self.is_bialgebra() and self.has_antipode() \
            and (self.R is None or self.is_quasitriangular())

    @classmethod
    def group_algebra(cls, table):
        """
        The group algebra :math:`k[G]` from a group multiplication ``table``,
        with ``table[i][j]`` the index of :math:`g_i g_j` and ``g_0`` the unit.

        >>> Z2 = HopfAlgebra.group_algebra([[0, 1], [1, 0]])
        >>> assert Z2.is_valid()
        """
        table = [list(row) for row in table]
        n = len(table)
        inverse = [next(j for j in range(n) if table[i][j] == 0)
                   for i in range(n)]
        unit = np.zeros(n)
        unit[0] = 1
        counit = np.ones(n)
        mult = np.zeros((n, n, n))
        comult = np.zeros((n, n, n))
        antipode = np.zeros((n, n))
        for i in range(n):
            comult[i, i, i] = 1                 # grouplike
            antipode[i, inverse[i]] = 1
            for j in range(n):
                mult[i, j, table[i][j]] = 1
        R = np.zeros((n, n))
        R[0, 0] = 1                             # cocommutative: trivial R
        return cls.from_arrays(unit, counit, mult, comult, antipode, R)

    @classmethod
    def cyclic(cls, n):
        """
        The group algebra of the cyclic group :math:`\\mathbb{Z}/n`.

        >>> assert HopfAlgebra.cyclic(3).is_valid()
        """
        table = [[(i + j) % n for j in range(n)] for i in range(n)]
        return cls.group_algebra(table)

    @classmethod
    def sweedler(cls):
        """
        Sweedler's four-dimensional Hopf algebra, the smallest one that is
        neither commutative nor cocommutative, with basis :math:`1, g, x, gx`
        (:math:`g^2 = 1`, :math:`x^2 = 0`, :math:`xg = -gx`) and
        :math:`S^2 \\neq \\mathrm{id}`.

        >>> H = HopfAlgebra.sweedler()
        >>> assert H.is_valid() and H.dim == 4
        >>> assert not H.is_commutative() and not H.is_cocommutative()
        """
        unit = np.array([1, 0, 0, 0])
        counit = np.array([1, 1, 0, 0])
        mult = np.zeros((4, 4, 4))
        for j in range(4):
            mult[0, j, j] = 1                      # 1 . e_j = e_j
        mult[1, 0, 1] = mult[1, 1, 0] = 1          # g.1=g, g.g=1
        mult[1, 2, 3] = mult[1, 3, 2] = 1          # g.x=gx, g.gx=x
        mult[2, 0, 2] = 1                          # x.1=x
        mult[2, 1, 3] = -1                         # x.g = -gx
        mult[3, 0, 3] = 1                          # gx.1=gx
        mult[3, 1, 2] = -1                         # gx.g = -x
        comult = np.zeros((4, 4, 4))
        comult[0, 0, 0] = 1                        # D(1) = 1 (x) 1
        comult[1, 1, 1] = 1                        # D(g) = g (x) g
        comult[2, 2, 0] = comult[2, 1, 2] = 1      # D(x) = x(x)1 + g(x)x
        comult[3, 3, 1] = comult[3, 0, 3] = 1      # D(gx) = gx(x)g + 1(x)gx
        antipode = np.zeros((4, 4))
        antipode[0, 0] = antipode[1, 1] = 1        # S(1)=1, S(g)=g
        antipode[2, 3] = -1                        # S(x) = -gx
        antipode[3, 2] = 1                         # S(gx) = x
        return cls.from_arrays(unit, counit, mult, comult, antipode)

    def double(self):
        """
        The Drinfeld quantum double :math:`D(H) = H \\otimes H^*`
        (Def. 1.23 of *Hopf Algebras in Quantum Computation*), a
        quasitriangular Hopf algebra of dimension ``self.dim ** 2`` whose
        generators are **composites** of ``self``'s generators, their
        ``.dagger()`` (the :math:`H^*` structure) and cups/caps — never
        materialised as big tensors.

        The coadjoint multiplication splits :math:`h` and :math:`\\phi'`
        each into three, pairs :math:`\\phi'_1` with :math:`S^{-1}(h_3)` and
        :math:`\\phi'_3` with :math:`h_1`, and multiplies
        :math:`\\phi \\phi'_2` in :math:`H^*` and :math:`h_2 h'` in
        :math:`H` — the ``order`` below routes the wires to
        :math:`\\phi'_1 h_3 \\phi'_3 h_1 \\phi \\phi'_2 h_2 h'`. The antipode
        is the anti-homomorphism :math:`(\\epsilon \\otimes S) \\circ
        ((S^{-1})^* \\otimes 1)` composed with that same multiplication.

        >>> D = HopfAlgebra.cyclic(2).double()
        >>> assert D.dim == 4 and D.is_valid() and D.is_quasitriangular()
        """
        ty = self.ty
        Sinv = self.antipode_inv
        mstar, dstar = self.comult.dagger(), self.mult.dagger()

        unit = self.counit.dagger() @ self.unit
        counit = self.unit.dagger() @ self.counit
        comult = dstar @ self.comult \
            >> Diagram.swap(ty, ty) @ ty @ ty >> ty @ Diagram.swap(ty, ty) @ ty
        R = self.counit.dagger() @ Diagram.caps(ty, ty) @ self.unit

        comult2 = self.comult >> self.comult @ ty
        cstar2 = dstar >> dstar @ ty
        split = Id(ty) @ comult2 @ cstar2 @ Id(ty)
        order = [4, 3, 6, 1, 0, 5, 2, 7]
        perm = [i for b in order
                for i in range(b * len(ty), b * len(ty) + len(ty))]
        route = Diagram.permutation(perm, split.cod)
        sinv = Id(ty) @ Sinv @ Id(ty ** 6)
        contract = Diagram.cups(ty, ty) @ Diagram.cups(ty, ty) \
            @ mstar @ self.mult
        mult = split >> route >> sinv >> contract

        prep = self.counit.dagger() \
            @ (Sinv.dagger() @ self.antipode) @ self.unit
        antipode = prep >> ty @ Diagram.swap(ty, ty) @ ty >> mult
        return HopfAlgebra(unit, counit, mult, comult, antipode, R=R)


class Representation(frobenius.Dim):
    """
    A finite-dimensional (left) module over a :class:`HopfAlgebra`, i.e. an
    object of :math:`\\mathrm{Rep}(H)` and of the category
    :class:`Intertwiner`. It is a :class:`.tensor.Dim` (its underlying vector
    space) carrying the ``algebra`` and its ``action`` diagram
    :math:`H \\otimes V \\to V` (with ``V = action.cod``), which the ribbon
    classmethods of :class:`Intertwiner` read off. Its adjoints :attr:`l` and
    :attr:`r` are the dual module :math:`V^*` with the antipode-twisted action
    :math:`\\rho^*(h) = \\rho(S h)^T`; a product of representations
    (:meth:`tensor`) drops the payload, leaving just a :class:`.tensor.Dim`.

    Parameters:
        inside : The dimensions of the underlying space (for a product).
        algebra : The :class:`HopfAlgebra`.
        action : The action diagram :math:`H \\otimes V \\to V`, so that
            ``V = action.cod``.

    A representation satisfies the two module axioms as diagram equations
    (:meth:`is_module`): the action is associative over ``mult`` and unital
    over ``unit``. Here is the associativity axiom
    ``(mult @ V) >> action == (H @ action) >> action`` for a direct sum of
    two anyon modules of :math:`D(\\mathbb{Z}/2)`, with the left-hand side
    drawn as the :class:`.tensor.CMap` it contracts to:

    >>> D = HopfAlgebra.cyclic(2).double()
    >>> e = Representation.anyon(D, 0, -1)
    >>> m = Representation.anyon(D, 1, 1)
    >>> V = Representation.direct_sum([e, m])
    >>> assert V.is_module() and V == Dim(2)
    >>> ty = V.action.cod
    >>> (D.mult @ ty >> V.action).to_map().draw(
    ...     path='docs/_static/hopf/module.png')

    .. image:: /_static/hopf/module.png
        :align: center
    """
    def __init__(self, *inside, algebra=None, action=None):
        self.algebra, self.action = algebra, action
        if action is not None and not inside:
            inside = action.cod.inside
        flat = [i for x in inside
                for i in (x.inside if isinstance(x, frobenius.Dim) else (x,))]
        super().__init__(*flat)

    def __repr__(self):
        if self.action is None:
            return factory_name(type(self)) \
                + f"({', '.join(map(repr, self.inside))})"
        return factory_name(type(self)) \
            + f"(algebra={self.algebra!r}, action={self.action!r})"

    def __eq__(self, other):
        """
        A payload-free representation is just a :class:`.tensor.Dim`, while
        two modules differ by their ``(algebra, action)`` — so distinct
        one-dimensional anyons are distinct objects.
        """
        if not (isinstance(other, frobenius.Dim)
                and self.inside == other.inside):
            return False
        if isinstance(other, Representation) \
                and self.action is not None and other.action is not None:
            return (self.algebra, self.action) == (other.algebra, other.action)
        return True

    def __hash__(self):
        return hash((self.inside, repr(self.algebra), repr(self.action)))

    @property
    def r(self):
        """
        The dual module :math:`V^*` with :math:`\\rho^*(h) = \\rho(S h)^T`,
        built diagrammatically: the antipode composed with the partial
        transpose of ``action``, taken with cups and caps. The legs of
        :math:`V^*` come out in reversed order. Left and right adjoints
        coincide, as for any :class:`.tensor.Dim`.

        >>> H = HopfAlgebra.sweedler()
        >>> assert Representation.regular(H).r.is_module()
        """
        if self.action is None:
            return self.ob(*self.inside[::-1])
        H, ty = self.algebra, self.action.cod
        hn, lv = len(H.ty), len(ty)
        twisted = H.antipode @ Id(ty) >> self.action
        bend = Id(H.ty @ ty.r) @ Diagram.caps(ty, ty.r)
        blocks = [hn, lv, lv, lv]
        starts = [sum(blocks[:i]) for i in range(len(blocks))]
        perm = [i for b in [0, 2, 1, 3]
                for i in range(starts[b], starts[b] + blocks[b])]
        contract = twisted @ Id(ty.r @ ty.r) \
            >> Diagram.cups(ty, ty.r) @ Id(ty.r)
        action = bend >> Diagram.permutation(perm, bend.cod) >> contract
        return type(self)(algebra=H, action=action)

    l = r

    def qdim(self):
        """
        The quantum dimension: the value of a loop coloured by ``V``, i.e. the
        (co)evaluation :math:`\\cup \\circ \\cap` in :class:`Intertwiner`.
        """
        loop = Intertwiner.caps(self, self.r) >> Intertwiner.cups(self, self.r)
        return complex(loop.eval(dtype=complex).array)

    def is_module(self):
        """
        Whether ``action`` is a representation: the two module axioms hold as
        diagram equations — the action is associative over ``mult`` and
        unital over ``unit``. See the axiom drawn in the class docstring.
        """
        H, ty = self.algebra, self.action.cod
        return all(
            lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
            for lhs, rhs in [
                (H.mult @ ty >> self.action,
                 H.ty @ self.action >> self.action),
                (H.unit @ ty >> self.action, Id(ty))])

    @classmethod
    def regular(cls, algebra):
        """ The regular representation, acting on ``H`` by ``mult``. """
        return cls(algebra=algebra, action=algebra.mult)

    @classmethod
    def anyon(cls, double, flux, charge):
        """
        A one-dimensional anyon module of the quantum double of a cyclic group
        algebra: the group element ``e_a`` acts by ``charge ** a`` in the flux
        sector ``flux``.
        """
        n = int(round(double.dim ** 0.5))
        assert n * n == double.dim, "not the double of an n-dim algebra"
        array = np.zeros((n, n), dtype=complex)
        for a in range(n):
            array[flux, a] = charge ** a
        action = Box[complex](
            'ρ', double.ty @ Dim(1), Dim(1), array.reshape(-1).tolist())
        return cls(algebra=double, action=action)

    @classmethod
    def direct_sum(cls, reps):
        """
        The direct sum of modules over one algebra, acting block-diagonally.

        >>> D = HopfAlgebra.cyclic(2).double()
        >>> e = Representation.anyon(D, 0, -1)
        >>> m = Representation.anyon(D, 1, 1)
        >>> V = Representation.direct_sum([e, m])
        >>> assert V.is_module() and V == Dim(2)
        """
        H = reps[0].algebra
        dims = [product(rep.inside) for rep in reps]
        d = sum(dims)
        array = np.zeros(H.ty.inside + (d, d), dtype=complex)
        offset = 0
        for rep, dim in zip(reps, dims):
            block = rep.action.eval(dtype=complex).array
            block = block.reshape(H.ty.inside + (dim, dim))
            block_slice = slice(offset, offset + dim)
            array[..., block_slice, block_slice] = block
            offset += dim
        action = Box[complex](
            'ρ', H.ty @ Dim(d), Dim(d), array.reshape(-1).tolist())
        return cls(algebra=H, action=action)


class Intertwiner(tensor.Diagram, RibbonCategory):
    """
    The ribbon category :math:`\\mathrm{Rep}(H)` of representations of a
    quasitriangular :class:`HopfAlgebra`: a category of
    :class:`.tensor.Diagram`\\ s whose objects are :class:`Representation`\\ s.
    Its ribbon structure is given by its classmethods — the braiding is
    :meth:`braid` (the R-matrix acting on the two strands, then a swap), the
    :meth:`twist` is the trace of the self-braiding, and the (co)evaluations
    ``cups``/``caps`` pair a module with its dual :attr:`Representation.r`,
    which carries the antipode-twisted action.

    Example
    -------
    An intertwiner ``f`` between modules is a map that commutes with the
    action, :math:`f \\circ \\rho_V = \\rho_W \\circ (1_H \\otimes f)`. We
    check this axiom for the braid on :math:`V \\otimes V`, whose action goes
    through the comultiplication:

    >>> import numpy as np
    >>> from discopy.drawing import Equation
    >>> D = HopfAlgebra.cyclic(2).double()
    >>> e = Representation.anyon(D, 0, -1)
    >>> m = Representation.anyon(D, 1, 1)
    >>> V = Representation.direct_sum([e, m])
    >>> ty = V.action.cod
    >>> action = D.comult @ Id(ty @ ty) \\
    ...     >> Id(D.ty) @ Diagram.swap(D.ty, ty) @ Id(ty) \\
    ...     >> V.action @ V.action
    >>> braid = Intertwiner.braid(V, V)
    >>> lhs, rhs = action >> braid, Id(D.ty) @ braid >> action
    >>> assert lhs.eval(dtype=complex).is_close(rhs.eval(dtype=complex))
    >>> Equation(lhs, rhs).draw(path='docs/_static/hopf/intertwiner.png')

    .. image:: /_static/hopf/intertwiner.png
        :align: center

    The braid contracts to the braiding matrix of the toric code:

    >>> matrix = braid.eval(dtype=complex).array.reshape(4, 4)
    >>> assert np.allclose(matrix, [[1, 0, 0, 0], [0, 0, -1, 0],
    ...                             [0, 1, 0, 0], [0, 0, 0, 1]])
    """
    ob = Representation

    @classmethod
    def braid(cls, left, right, is_dagger=False):
        """
        The braiding :math:`V \\otimes W \\to W \\otimes V` (its inverse
        :math:`R^{-1} = (S \\otimes 1) R` when ``is_dagger``): the R-matrix
        acting on the two strands, i.e. :math:`(\\rho_V \\otimes \\rho_W)(R)`,
        then a swap. Raises a :class:`ValueError` if the algebra has no
        R-matrix.
        """
        H = left.algebra
        if H is None or H.R is None:
            raise ValueError("the braiding needs a quasitriangular structure")
        hn = len(H.ty)

        def r_action(a, b, r_matrix):
            tya, tyb = Dim(*a.inside), Dim(*b.inside)
            la, lb = len(tya), len(tyb)
            arrows = r_matrix @ Id(tya @ tyb)
            order = list(range(hn)) + list(range(2 * hn, 2 * hn + la)) \
                + list(range(hn, 2 * hn)) \
                + list(range(2 * hn + la, 2 * hn + la + lb))
            return arrows >> Diagram.permutation(order, arrows.cod) \
                >> a.action @ b.action

        swap = Diagram.swap(Dim(*left.inside), Dim(*right.inside))
        if is_dagger:
            body = swap >> r_action(right, left, H.R >> H.antipode @ Id(H.ty))
        else:
            body = r_action(left, right, H.R) >> swap
        return cls(body.inside, body.dom, body.cod)

    @classmethod
    def cups(cls, left, right):
        """ The evaluation of a module against its dual, the plain pairing. """
        body = Diagram.cups(Dim(*left.inside), Dim(*right.inside))
        return cls(body.inside, body.dom, body.cod)

    @classmethod
    def caps(cls, left, right):
        """ The coevaluation of a module against its dual. """
        body = Diagram.caps(Dim(*left.inside), Dim(*right.inside))
        return cls(body.inside, body.dom, body.cod)

    @classmethod
    def twist(cls, dom):
        """ The twist of ``dom``: the ribbon trace of its self-braiding. """
        return cls.id(dom) @ cls.caps(dom, dom.r) \
            >> cls.braid(dom, dom) @ cls.id(dom.r) \
            >> cls.id(dom) @ cls.cups(dom, dom.r)


class Functor(ribbon.Functor):
    """
    A ribbon functor from :mod:`.ribbon` diagrams to :class:`Intertwiner`, i.e.
    it sends a knot to the **tensor network** of :math:`\\mathrm{Rep}(H)` that
    computes its invariant. Contract it yourself with
    :meth:`.tensor.Diagram.eval`.

    Parameters:
        ob : A mapping from atomic :class:`.ribbon.Ty` to
            :class:`Representation`.
        ar : A mapping from generating :class:`.ribbon.Box` to arrays.

    The ribbon structure of :math:`\\mathrm{Rep}(H)` lives in the codomain
    :class:`Intertwiner`: each generator is routed to a classmethod. A
    cup/cap/twist goes to ``Intertwiner.cups``/``caps``/
    :meth:`~Intertwiner.twist` by the inherited dispatch, and both crossings of
    a braid to :meth:`Intertwiner.braid` (the under-crossing being the antipode
    inverse, not the tensor adjoint).

    Example
    -------
    The functor sends a :class:`.ribbon.Braid` to the tensor network of the
    braiding — the R-matrix acting on the two strands, then a swap:

    >>> D = HopfAlgebra.cyclic(2).double()
    >>> e = Representation.anyon(D, 0, -1)
    >>> m = Representation.anyon(D, 1, 1)
    >>> V = Representation.direct_sum([e, m])
    >>> x = ribbon.Ty('x')
    >>> F = Functor(ob_map={x: V}, ar_map={})
    >>> network = F(ribbon.Braid(x, x))
    >>> from discopy import tensor
    >>> assert isinstance(network, tensor.Diagram)
    >>> network.draw(path='docs/_static/hopf/braid_network.png')

    .. image:: /_static/hopf/braid_network.png
        :align: center

    The user contracts it to a :class:`.tensor.Tensor`:

    >>> import numpy as np
    >>> matrix = network.eval(dtype=complex).array.reshape(4, 4)
    >>> assert np.allclose(matrix, [[1, 0, 0, 0], [0, 0, -1, 0],
    ...                             [0, 1, 0, 0], [0, 0, 0, 1]])
    """
    dom, cod = ribbon.Diagram, Intertwiner

    def __call__(self, other):
        """
        On objects, an atom goes to its :class:`Representation` — the dual
        :attr:`Representation.r` when the winding is odd — and a product of
        atoms to the product of their dimensions. On arrows, both crossings
        of a braid go to :meth:`Intertwiner.braid`, generating boxes to their
        arrays, and everything else through the inherited ribbon dispatch.
        """
        if isinstance(other, ribbon.Ty):
            reps = [self.ob_map[ribbon.Ty(ob.name)].r if ob.z % 2
                    else self.ob_map[ribbon.Ty(ob.name)]
                    for ob in other.inside]
            return reps[0] if len(reps) == 1 else Dim().tensor(*reps)
        if isinstance(other, ribbon.Braid):
            return self.cod.braid(self(other.dom[:1]), self(other.dom[1:]),
                                  is_dagger=other.is_dagger)
        if isinstance(other, ribbon.Box) and not isinstance(
                other, (ribbon.Cup, ribbon.Cap, ribbon.Twist)):
            return Box(other.name, self(other.dom), self(other.cod),
                       self.ar_map[other])
        return super().__call__(other)
