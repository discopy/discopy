# -*- coding: utf-8 -*-

"""
The equations a generator promises, and how strongly a learned module keeps
them.

:mod:`discopy.neural.signature` says *which* ports of a box are one orbit
under which group; this module says what that **means**, and measures it.
A symmetry is a group action

.. math:: \\rho_X : G \\to \\mathrm{Aut}(X)

and the promise a module makes at a site is a commuting law, i.e.
equivariance:

.. math:: F \\circ \\rho_X(g) = \\rho_Y(g) \\circ F \\quad
          \\text{for every } g \\in G.

For a cell of :mod:`discopy.neural.cells` the domain and the codomain of
:math:`F` are the *same* object -- the boundary of the box, see
:class:`~discopy.neural.map.InteractionMap` -- so :math:`\\rho_X = \\rho_Y`
and the law reads :math:`F \\circ \\rho(g) = \\rho(g) \\circ F`.

How strongly a law holds is a :class:`Strictness`, and being honest about
it is the point:

* **strict** -- holds by construction.  Swaps, cups, caps and traces are
  wiring in the target category, so a functor into :mod:`discopy.neural`
  preserves them exactly and for free.
* **lax** -- holds up to the reordering of a floating-point reduction.
  That is what a pooled cell gives: :class:`~discopy.neural.cells.Relation`
  and :class:`~discopy.neural.cells.Site` are permutation-equivariant
  because they pool symmetrically, and the residual
  :func:`check_equivariant` reports is rounding error, not error.
* **approximate** -- neither, only measured or regularised.

Note
----
Permutation equivariance is **not** Frobenius structure.  A learned
:class:`~discopy.neural.cells.Relation` commutes with permutations of its
members and still does not fuse: :func:`fusion_residual` measures how far
it is from the fusion law, and the answer is not zero.  Satisfying the
symmetry a signature declares says nothing about the other equations of
whatever algebraic theory one had in mind.

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
from typing import Mapping

from discopy.neural.signature import (
    Orbit, Signature, Sym, leg_generators)
from discopy.utils import AxiomError


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

    The group is named by a :class:`~discopy.neural.Sym` and its generators
    are :func:`~discopy.neural.signature.leg_generators`, so this is a
    *reading* of the signature rather than a second definition of it.

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
    >>> symmetry(Signature((Orbit(peer, 2), Orbit(state, traced=True))))
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


# --- the executable diagnostics --------------------------------------------

def check_equivariant(module, signature: Signature, widths: Mapping,
                      atol: float = 1e-5, batch: int = 4,
                      seed: int = 0) -> dict:
    """
    Measure whether a module satisfies the equations its signature
    declares, and refuse it when it does not.

    For each generator of the symmetry group, the module is run on a
    permuted input and compared against the permutation of its output on
    the original one.  A learned module is only ever *approximately*
    equivariant -- pooling reorders a floating-point reduction -- so the
    residual is reported rather than claimed to be zero.

    Parameters:
        module : The torch module filling the site.
        signature : The signature the site declares.
        widths : The width carried by each atomic role.
        atol : The residual above which the module is rejected.
        batch : The number of random rows to test on.
        seed : The seed of the random input, so the check is reproducible.

    Returns:
        The largest residual per role, over every generator of its orbit.

    Raises:
        AxiomError : If a residual exceeds ``atol``.

    Example
    -------
    >>> import torch  # doctest: +EXTRA
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> message = Ty("message")
    >>> unit = Signature((Orbit(message, 3, Sym.PERM), ))
    >>> class Mean(torch.nn.Module):
    ...     def forward(self, x):
    ...         return x.reshape(-1, 3, 2).mean(1, keepdim=True).expand(
    ...             -1, 3, -1).reshape(-1, 6)
    >>> check_equivariant(Mean(), unit, {message: 2})[message] < 1e-6
    True
    >>> class Skew(torch.nn.Module):
    ...     def forward(self, x):
    ...         return x * torch.arange(1., 1 + x.shape[-1], dtype=x.dtype)
    >>> try:
    ...     check_equivariant(Skew(), unit, {message: 2})
    ... except AxiomError as error:
    ...     print(str(error).split(":")[0])
    message is not perm-equivariant
    """
    import torch
    total = signature.width(widths)
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(batch, total, generator=generator, dtype=torch.double)
    residuals: dict = {}
    with torch.no_grad():
        expected = module(rows)
        for orbit, permutation in _orbit_generators(signature):
            index = _flat_index(signature, widths, permutation)
            residual = float(
                (module(rows[:, index]) - expected[:, index]).abs().max())
            for atom in orbit.role:
                residuals[atom] = max(residuals.get(atom, 0.0), residual)
    broken = {atom: value for atom, value in residuals.items()
              if value > atol}
    if broken:
        raise AxiomError(", ".join(
            f"{atom} is not {_sym_of(signature, atom)}-equivariant: "
            f"residual {value:.3g} > {atol:g}"
            for atom, value in broken.items()))
    return residuals


def fusion_residual(module, signature: Signature, widths: Mapping,
                    arity: int = 4, rounds: int = 4, batch: int = 4,
                    seed: int = 0) -> float:
    """
    How far a learned relation is from fusing, i.e. from being a spider.

    The Frobenius fusion law says two spiders wired along a leg are one
    spider over the remaining legs.  Here the two relations are glued
    along their last leg and message passing is run on the composite until
    the shared wire settles; the free legs are then compared against one
    relation over all of them.  A learned module does **not** satisfy the
    law -- it is lax, not strict -- so this is the number that says so,
    reported rather than a docstring claiming a structure the weights do
    not have.

    Parameters:
        module : The relation to measure.
        signature : Its signature, one orbit of members.
        widths : The width each atomic role carries.
        arity : The number of free legs on each side of the gluing.
        rounds : The exchanges along the shared wire.
        batch : The number of random rows to measure on.
        seed : The seed of the random input.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Orbit, Signature, Sym
    >>> from discopy.neural.cells import Relation  # doctest: +EXTRA
    >>> message = Ty("message")
    >>> unit = Signature((Orbit(message, 4, Sym.PERM), ))
    >>> box = Relation(unit, {message: 2}, hidden=4).double()
    >>> fusion_residual(box, unit, {message: 2}) > 1e-3
    True
    """
    import torch
    leg = sum(widths[atom] for atom in signature.orbits[0].role)
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(batch, 2 * arity * leg, generator=generator,
                       dtype=torch.double)
    left, right = rows[:, :arity * leg], rows[:, arity * leg:]
    shared = torch.zeros(batch, 2 * leg, dtype=rows.dtype)
    with torch.no_grad():
        for _ in range(rounds):
            one = module(torch.cat([left, shared[:, leg:]], -1))
            other = module(torch.cat([right, shared[:, :leg]], -1))
            shared = torch.cat([one[:, -leg:], other[:, -leg:]], -1)
        glued = torch.cat([one[:, :-leg], other[:, :-leg]], -1)
        return float((glued - module(rows)).abs().max())


def _orbit_generators(signature: Signature):
    """ Each generator of :meth:`Signature.generators` with its orbit. """
    result, generators = [], iter(signature.generators())
    for orbit in signature.orbits:
        for _ in leg_generators(orbit.sym, orbit.arity):
            result.append((orbit, next(generators)))
    return result


def _sym_of(signature: Signature, role) -> Sym:
    """ The symmetry of the orbit an atomic role belongs to. """
    for orbit in signature.orbits:
        if role in tuple(orbit.role):
            return orbit.sym
    raise KeyError(role)


def _flat_index(signature: Signature, widths: Mapping, permutation):
    """ A port permutation as a permutation of the flat message vector. """
    import torch
    port_widths = [widths[role] for role in signature.roles]
    offsets, total = [], 0
    for width in port_widths:
        offsets.append(total)
        total += width
    return torch.tensor([
        k for port in permutation.inside
        for k in range(offsets[port], offsets[port] + port_widths[port])],
        dtype=torch.long)
