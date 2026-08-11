# -*- coding: utf-8 -*-

"""
The equations a site promises, read as group actions.

:mod:`discopy.neural.signature` says *which* ports of a box are one orbit
under which group; this module says what that **means**, and how strongly.
A symmetry is a group action

.. math:: \\rho_X : G \\to \\mathrm{Aut}(X)

and the promise a module makes at a site is a commuting law, i.e.
equivariance:

.. math:: F \\circ \\rho_X(g) = \\rho_Y(g) \\circ F \\quad
          \\text{for every } g \\in G.

For a cell of :mod:`discopy.neural.cells` the domain and the codomain of
:math:`F` are the *same* object -- the boundary of the box, see
:mod:`discopy.neural.parametric` -- so :math:`\\rho_X = \\rho_Y` and the law
reads :math:`F \\circ \\rho(g) = \\rho(g) \\circ F`.

Nothing here computes: :meth:`~discopy.neural.signature.Signature.
generators` is the same group acting on ports, and
:func:`~discopy.neural.signature.check_equivariant` is the executable
diagnostic that measures the residual of the law on an actual module.
This module is where the residual gets a name.

How strongly a law holds is a :class:`Strictness`, and being honest about
it is the point:

* **strict** -- holds by construction.  Swaps, cups, caps and traces are
  wiring in the target category, so a functor into :mod:`discopy.neural`
  preserves them exactly and for free.
* **lax** -- holds up to the reordering of a floating-point reduction.
  That is what a pooled cell gives: :class:`~discopy.neural.cells.Relation`
  and :class:`~discopy.neural.cells.Site` are permutation-equivariant
  because they pool symmetrically, and the residual
  :func:`~discopy.neural.signature.check_equivariant` reports is rounding
  error, not error.
* **approximate** -- neither, only measured or regularised.

Note
----
Permutation equivariance is **not** Frobenius structure.  A learned
:class:`~discopy.neural.cells.Relation` commutes with permutations of its
members and still does not fuse:
:func:`~discopy.neural.cells.fusion_residual` measures how far it is from
the fusion law, and the answer is not zero.  Satisfying the symmetry a
signature declares says nothing about the other equations of whatever
algebraic theory one had in mind.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Strictness
    Action
    Law

Example
-------

>>> from discopy.frobenius import Ty
>>> from discopy.neural import Orbit, Signature, Sym
>>> message, state = Ty("message"), Ty("state")
>>> unit = Signature((Orbit(message, 3, Sym.PERM), Orbit(state, traced=True)))
>>> for law in symmetry(unit):
...     print(law)
message is lax perm-equivariant over 3 legs
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import factorial

from discopy.neural.signature import Orbit, Signature, Sym, leg_generators


class Strictness(StrEnum):
    """
    How strongly a law is claimed to hold.

    * :attr:`STRICT` : by construction, exactly.
    * :attr:`LAX` : up to the reordering of a floating-point reduction.
    * :attr:`APPROXIMATE` : only measured or regularised.
    """

    STRICT = "strict"
    LAX = "lax"
    APPROXIMATE = "approximate"


@dataclass(frozen=True)
class Action:
    """
    A group action :math:`\\rho : G \\to \\mathrm{Aut}(X)` on the legs of an
    orbit, given by the images of a set of generators of :math:`G`.

    The group is named by a :class:`~discopy.neural.signature.Sym` and its
    generators are :func:`~discopy.neural.signature.leg_generators`, so this
    is a *reading* of the signature rather than a second definition of it.

    Parameters:
        group : The group acting.
        degree : The number of legs it acts on.

    Example
    -------
    >>> from discopy.neural import Sym
    >>> Action(Sym.PERM, 3).generators
    ((1, 0, 2), (1, 2, 0))
    >>> Action(Sym.CYCLIC, 4).order, Action(Sym.PERM, 4).order
    (4, 24)
    >>> Action(Sym.NONE, 4).is_trivial, Action(Sym.PERM, 1).is_trivial
    (True, True)
    """

    group: Sym
    degree: int

    @property
    def generators(self) -> tuple[tuple[int, ...], ...]:
        """ The images of the generators of the group, as permutations. """
        return tuple(leg_generators(self.group, self.degree))

    @property
    def is_trivial(self) -> bool:
        """
        Whether the action is trivial, i.e. whether the group has no
        generator here and the law says nothing.
        """
        return not self.generators

    @property
    def order(self) -> int:
        """
        The order of the group as it acts here: one when trivial,
        ``degree`` for a cyclic orbit, ``degree!`` for a symmetric one.
        """
        if self.is_trivial:
            return 1
        return self.degree if self.group == Sym.CYCLIC \
            else factorial(self.degree)


@dataclass(frozen=True)
class Law:
    """
    An equivariance law: :math:`F \\circ \\rho(g) = \\rho(g) \\circ F` for
    every generator :math:`g` of the group of :attr:`action`.

    Parameters:
        roles : The atomic roles one leg of the orbit carries.
        action : The group acting on its legs.
        strictness : How strongly the law is claimed to hold.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Sym
    >>> print(Law((Ty("peer"), ), Action(Sym.CYCLIC, 5), Strictness.STRICT))
    peer is strict cyclic-equivariant over 5 legs
    """

    roles: tuple
    action: Action
    strictness: Strictness = Strictness.LAX

    def __str__(self) -> str:
        return (f"{' @ '.join(map(str, self.roles))} is "
                f"{self.strictness} {self.action.group}-equivariant over "
                f"{self.action.degree} legs")


def action(orbit: Orbit) -> Action:
    """
    The action an orbit declares on its legs.

    Parameters:
        orbit : The orbit of ports.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Sym
    >>> action(Orbit(Ty("message"), 9, Sym.PERM))
    Action(group=<Sym.PERM: 'perm'>, degree=9)
    """
    return Action(orbit.sym, orbit.arity)


def symmetry(signature: Signature,
             strictness: Strictness = Strictness.LAX) -> tuple[Law, ...]:
    """
    The laws a signature declares: one per orbit whose action is not
    trivial, in orbit order.

    A traced orbit is not listed twice: the action permutes the legs and
    every copy of a leg alike, so one law covers both ends of a loop.

    Parameters:
        signature : The signature of a box.
        strictness : How strongly the laws are claimed to hold; ``LAX`` by
                     default, which is what a pooled cell gives.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> peer, state = Ty("peer"), Ty("state")
    >>> plain = Signature((Orbit(peer, 2), Orbit(state, traced=True)))
    >>> symmetry(plain)
    ()
    >>> clique = Signature((Orbit(peer, 4, Sym.PERM),
    ...                     Orbit(state, traced=True)))
    >>> [str(law) for law in symmetry(clique)]
    ['peer is lax perm-equivariant over 4 legs']

    Counting the generators of the laws counts the generators of the
    signature, which is the port-level realisation of the same group:

    >>> sum(len(law.action.generators) for law in symmetry(clique)) \\
    ...     == len(clique.generators())
    True
    """
    return tuple(
        Law(tuple(orbit.role), action(orbit), strictness)
        for orbit in signature.orbits if not action(orbit).is_trivial)
