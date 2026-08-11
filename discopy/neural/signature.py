# -*- coding: utf-8 -*-

"""
What a box promises: its ports, grouped into orbits, and the symmetry each
orbit carries.

A functor into :mod:`discopy.neural` preserves swaps, cups, caps and traces
strictly and for free, because they are wiring: a permutation of a flat
tensor. What it cannot preserve for free is a box whose legs carry a
symmetry -- a spider, a braid, a constraint unit over nine members. Those
stay boxes, and their equations hold **iff the torch module satisfies
them**. A :class:`Signature` is where that promise is written down: it says
how many ports a box has, which of them are one orbit under a group, and
which are traced; :func:`check_equivariant` then measures whether the
module keeps the promise.

The target category is compact closed and its port order is fixed --
:meth:`discopy.neural.CMap.box_ports` reads the domain then the reversed
codomain -- so the target is already the least symmetric case,
:attr:`Sym.NONE`. A signature never *adds* symmetry to the map; it
restricts which modules are admissible at a site.

A signature is also the single source of truth for the port layout of a
box. :meth:`Signature.cod` builds the abstract type, :meth:`Signature.loops`
gives the traced pairs the skeleton wires, and :meth:`Signature.slices`
gives the flat offsets the module reads and writes, so the three can no
longer disagree.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sym
    Orbit
    Signature

Example
-------

>>> from discopy.frobenius import Ty
>>> message, state, given = Ty("message"), Ty("state"), Ty("given")
>>> cell = Signature((
...     Orbit(message, 3, Sym.PERM), Orbit(state, traced=True),
...     Orbit(given, traced=True)))
>>> print(cell.cod)
message @ message @ message @ state @ state @ given @ given
>>> cell.positions(state)
(3, 4)
>>> cell.loops()
((3, 4), (5, 6))
>>> places = cell.slices({message: 24, state: 96, given: 24})
>>> {str(role): (block.start, block.stop) for role, block in places.items()}
{'message': (0, 72), 'state': (72, 168), 'given': (264, 288)}
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Mapping

from discopy import frobenius
from discopy.python.finset import Permutation
from discopy.utils import AxiomError


class Sym(StrEnum):
    """
    The symmetry an orbit of ports carries, i.e. the group the module at
    that site must be equivariant under.

    * :attr:`NONE` : the ports are distinguishable, no equation.
    * :attr:`PERM` : the symmetric group, e.g. the members of a constraint
      unit or the legs of a spider.
    * :attr:`CYCLIC` : the cyclic group, e.g. the legs of a planar node.
    """

    NONE = "none"
    PERM = "perm"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class Orbit:
    """
    A family of ports playing the same role.

    An orbit has ``arity`` legs, each carrying the (possibly composite)
    type ``role``; a traced orbit has each leg twice, the outgoing copy
    followed by the incoming one, which :meth:`Signature.loops` wires
    together.

    A composite ``role`` is one leg carrying several roles at once, which
    is how a recurrent cell with two states -- the ``h`` and ``c`` of an
    ``LSTMCell`` -- keeps them as two named roles on one loop rather than
    as two halves of one wide port.

    Parameters:
        role : The type of one leg, a product of atomic roles.
        arity : The number of legs.
        sym : The group the legs are an orbit under.
        traced : Whether each leg is a self-wired pair of ports.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> hidden, memory = Ty("hidden"), Ty("memory")
    >>> print(Orbit(hidden @ memory, traced=True).cod)
    hidden @ memory @ hidden @ memory
    """

    role: frobenius.Ty
    arity: int = 1
    sym: Sym = Sym.NONE
    traced: bool = False

    def __post_init__(self):
        if self.arity < 0:
            raise ValueError(self.arity)
        if len(self.role) > 1 and self.arity != 1:
            raise ValueError(
                "a leg carrying several roles cannot also be repeated")

    @property
    def copies(self) -> int:
        """ How many times each leg appears: two when traced, one else. """
        return 2 if self.traced else 1

    @property
    def cod(self) -> frobenius.Ty:
        """ The ports of the orbit, as a type. """
        return self.role ** (self.arity * self.copies)

    @property
    def n_ports(self) -> int:
        """ The number of ports of the orbit. """
        return len(self.role) * self.arity * self.copies


@dataclass(frozen=True)
class Signature:
    """
    The ports of a box, as a tuple of orbits, in logical port order.

    Parameters:
        orbits : The orbits, in the order their ports appear.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> peer, hidden, memory = Ty("peer"), Ty("hidden"), Ty("memory")
    >>> clique = Signature((
    ...     Orbit(peer, 4, Sym.PERM), Orbit(hidden @ memory, traced=True),
    ...     Orbit(Ty("given"), traced=True)))
    >>> clique.positions(hidden), clique.positions(memory)
    ((4, 6), (5, 7))
    >>> clique.loops()
    ((4, 6), (5, 7), (8, 9))
    """

    orbits: tuple[Orbit, ...]

    @property
    def cod(self) -> frobenius.Ty:
        """ The codomain of the abstract box: every port, in order. """
        result = frobenius.Ty()
        for orbit in self.orbits:
            result = result @ orbit.cod
        return result

    @property
    def roles(self) -> tuple[frobenius.Ty, ...]:
        """ The atomic role of each port, in logical port order. """
        return tuple(self.cod)

    def box(self, name: str, category=frobenius):
        """
        The abstract box of this signature: no domain, one port per role.

        The source category is a parameter, so the same signature builds a
        symmetric, compact or frobenius box; its ``require_planar``,
        ``require_acyclic``, ``require_oriented`` and ``require_connected``
        flags then do the guarding when the box is wired into a map.

        Parameters:
            name : The name of the box, which is also the key its module is
                   looked up under.
            category : The module the box and its types come from.

        Example
        -------
        >>> from discopy import compact, frobenius, symmetric
        >>> unit = Signature((Orbit(frobenius.Ty("message"), 3, Sym.PERM), ))
        >>> print(unit.box("unit").cod)
        message @ message @ message
        >>> [type(unit.box("unit", category=source)).__name__
        ...  for source in (frobenius, compact, symmetric)]
        ['Box', 'Box', 'Box']
        >>> isinstance(unit.box("unit", category=symmetric), symmetric.Box)
        True
        """
        typ = category.Ty(*[atom.inside[0].name for atom in self.cod])
        return category.Box(name, category.Ty(), typ)

    def positions(self, role: frobenius.Ty) -> tuple[int, ...]:
        """
        Where an atomic role sits in the logical port order.

        For a traced orbit the outgoing copies come first, so the ``i``-th
        and the ``arity + i``-th entries are the two ends of one loop.

        Parameters:
            role : The atomic role to locate.
        """
        return tuple(i for i, other in enumerate(self.roles)
                     if other == role)

    def heads(self, role: frobenius.Ty) -> tuple[int, ...]:
        """
        The positions of a role that a module *reads*: the outgoing copy
        of a traced orbit, every leg of an untraced one.

        Parameters:
            role : The atomic role to locate.
        """
        found = self.positions(role)
        for orbit in self.orbits:
            if role in tuple(orbit.role):
                return found[:len(found) // orbit.copies]
        raise KeyError(role)

    def loops(self) -> tuple[tuple[int, int], ...]:
        """
        The traced pairs of ports, as positions in the logical port order.

        Two readings of one pair, worth keeping apart.  *Structurally* a
        self-wired pair is the categorical trace of the compact target --
        :mod:`discopy.neural.skeleton` checks that equation as a doctest --
        and a functor into :mod:`discopy.neural` preserves it strictly,
        because it is wiring rather than a box.  *Dynamically* it is a
        persistent state channel: what a box writes on one end it reads
        back on the other one round later, so a value survives a round.
        That is delayed feedback under finite iteration, not a fixed point;
        see :mod:`discopy.neural.dynamics`.
        """
        result, cursor = [], 0
        for orbit in self.orbits:
            span = len(orbit.role) * orbit.arity
            if orbit.traced:
                result += [(cursor + i, cursor + span + i)
                           for i in range(span)]
            cursor += span * orbit.copies
        return tuple(result)

    def slices(self, widths: Mapping) -> dict:
        """
        Where each atomic role sits in the flat message vector of a box,
        given the width of every role.

        This is the one place a port offset is computed. A module reads and
        writes through these slices, so its cursor arithmetic and the type
        of its abstract box can no longer disagree.

        Parameters:
            widths : The width carried by each atomic role; roles of width
                     zero are erased, exactly as ``Dim(0)`` erases a port.
        """
        result, cursor = {}, 0
        for orbit in self.orbits:
            leg = sum(widths[atom] for atom in orbit.role)
            block = orbit.arity * leg
            inner = cursor
            for atom in orbit.role:
                width = widths[atom]
                if width:
                    result[atom] = slice(
                        inner, inner + (block if len(orbit.role) == 1
                                        else width))
                inner += width
            cursor += orbit.copies * block
        return result

    def width(self, widths: Mapping) -> int:
        """ The total flat width of a box under the given role widths. """
        return sum(
            orbit.copies * orbit.arity
            * sum(widths[atom] for atom in orbit.role)
            for orbit in self.orbits)

    def resize(self, role: frobenius.Ty, arity: int) -> Signature:
        """
        The same signature with the arity of one orbit changed, which is
        how one shared module serves sites of different degree.

        Parameters:
            role : The role of the orbit to resize.
            arity : Its new arity.
        """
        return Signature(tuple(
            replace(orbit, arity=arity) if role in tuple(orbit.role)
            else orbit for orbit in self.orbits))

    def generators(self) -> list[Permutation]:
        """
        The generators of the symmetry group of the signature, as
        permutations of its ports.

        A permutation acts on the *legs* of one orbit and on every copy of
        each leg alike, so a traced orbit stays traced. The identity of
        the group generated is the equation the module at this site must
        satisfy; :func:`check_equivariant` measures how far it is from it,
        and :func:`discopy.neural.laws.symmetry` reads the same data as a
        group action and an equivariance law.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> unit = Signature((Orbit(Ty("message"), 3, Sym.PERM), ))
        >>> [tuple(permutation.inside) for permutation in unit.generators()]
        [(1, 0, 2), (1, 2, 0)]
        """
        result, cursor = [], 0
        for orbit in self.orbits:
            span, legs = len(orbit.role), orbit.arity
            block = span * legs
            for cycle in leg_generators(orbit.sym, legs):
                mapping = list(range(len(self.roles)))
                for copy in range(orbit.copies):
                    start = cursor + copy * block
                    for leg in range(legs):
                        for atom in range(span):
                            mapping[start + leg * span + atom] = \
                                start + cycle[leg] * span + atom
                result.append(Permutation(mapping))
            cursor += block * orbit.copies
        return result


def leg_generators(sym: Sym, arity: int) -> list[tuple[int, ...]]:
    """
    The generators of a symmetry group, as permutations of legs.

    This is the group itself, before it acts on anything:
    :meth:`Signature.generators` is the same group acting on ports and
    :mod:`discopy.neural.laws` is where it is read as an action
    :math:`\\rho : G \\to \\mathrm{Aut}(X)`.

    Parameters:
        sym : The symmetry the legs carry.
        arity : The number of legs.

    Example
    -------
    >>> leg_generators(Sym.PERM, 3)
    [(1, 0, 2), (1, 2, 0)]
    >>> leg_generators(Sym.CYCLIC, 3)
    [(1, 2, 0)]
    >>> leg_generators(Sym.NONE, 3), leg_generators(Sym.PERM, 1)
    ([], [])
    """
    if arity < 2 or sym == Sym.NONE:
        return []
    rotation = tuple(range(1, arity)) + (0, )
    if sym == Sym.CYCLIC:
        return [rotation]
    swap = (1, 0) + tuple(range(2, arity))
    return [swap, rotation]


def check_equivariant(module, signature: Signature, widths: Mapping,
                      atol: float = 1e-5, batch: int = 4,
                      seed: int = 0) -> dict:
    """
    Measure whether a module satisfies the equations its signature
    declares, and refuse it when it does not.

    For each generator of the symmetry group, the module is run on a
    permuted input and compared against the permutation of its output on
    the original one. A learned module is only ever *approximately*
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
    >>> import torch
    >>> from discopy.frobenius import Ty
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


def _flat_index(signature: Signature, widths: Mapping,
                permutation: Permutation):
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
