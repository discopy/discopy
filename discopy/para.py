# -*- coding: utf-8 -*-

"""
The category of parametric maps, Para.

Para is a construction that turns a symmetric monoidal category :math:`\\mathcal{C}` into a
category :math:`\\mathbf{Para}(\\mathcal{C})` where morphisms are equipped with parameter space.
A morphism in :math:`\\mathbf{Para}(\\mathcal{C})` from :math:`A` to :math:`B` with parameters
:math:`P` is a morphism :math:`A \\otimes P \\to B` in :math:`\\mathcal{C}`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Reparam
    Category

Axioms
------

* Sequential composition :math:`(P, f) >> (Q, g) = (P \\otimes Q, (f \\otimes \\mathrm{id}_Q) >> g)`

>>> from discopy.symmetric import Ty
>>> a, b, c = map(Ty, "ABC")
>>> p, q = map(Ty, "PQ")
>>> f = Box("f", a, b, p)
>>> g = Box("g", b, c, q)
>>> assert (f >> g).params == p @ q
>>> assert (f >> g).dom == a
>>> assert (f >> g).cod == c
>>> assert (f >> g).inside.dom == a @ p @ q

* Parallel composition :math:`(P, f) \\space @ \\space (Q, g) = (P \\otimes Q, (\\mathrm{id}_A \\otimes \\mathrm{swap}_{C, P} \\otimes \\mathrm{id}_Q) >> (f \\otimes g))`

>>> d = Ty('D')
>>> h = Box("h", c, d, q)
>>> assert (f @ h).params == p @ q
>>> assert (f @ h).dom == a @ c
>>> assert (f @ h).cod == b @ d
>>> assert (f @ h).inside.dom == a @ c @ p @ q

* Reparameterization :math:`f\\mathrm{.reparam}(r1 >> r2) == f\\mathrm{.reparam}(r2)\\mathrm{.reparam}(r1)`

>>> p_prime, p_prime_prime = Ty("P'"), Ty("P''")
>>> r2 = Box("r2", p_prime, p)
>>> r1 = Box("r1", p_prime_prime, p_prime)
>>> assert f.reparam(r1 >> r2) == f.reparam(r2).reparam(r1)

References
----------

* :cite:t:`Gavranovic24` Gavranovic, B., 2024.
  Fundamental Components of Deep Learning: A Category-Theoretic Approach.
* :cite:t:`GavranovicEtAl24` Gavranovic, B. et al., 2024.
  Position: Categorical Deep Learning is an Algebraic Theory of All Architectures.
"""

from __future__ import annotations

from discopy import monoidal, symmetric
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.utils import assert_isinstance, factory_name


def _as_para_diagram(diagram):
    """Map a symmetric monoidal diagram into a Para diagram with empty parameters."""
    if isinstance(diagram, Diagram):
        return diagram
    result = Diagram(diagram.inside, diagram.dom, diagram.cod)
    result._params = Ty()
    result._data_dom = diagram.dom
    result._data_cod = diagram.cod
    return result


def _as_monoidal_diagram(diagram):
    """Strip Para metadata and keep only the underlying symmetric diagram."""
    if isinstance(diagram, symmetric.Diagram):
        return symmetric.Diagram(diagram.inside, diagram.dom, diagram.cod)
    if isinstance(diagram, monoidal.Diagram):
        return symmetric.Diagram(diagram.inside, diagram.dom, diagram.cod)
    return diagram


@factory
class Diagram(symmetric.Diagram):
    """
    A Para diagram is a morphism in the base category with a parameter space.

    Parameters:
        inside (category.ar) : The underlying morphism :math:`A \\otimes P \\to B`.
        dom (category.ob) : The data domain :math:`A`.
        cod (category.ob) : The data codomain :math:`B`.
        params (category.ob) : The parameter space :math:`P`.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            swap
            copy
            reparameterize
            draw
            to_drawing
    """
    ty_factory = Ty

    def __init__(self, inside, dom, cod, _scan=True):
        super().__init__(inside, dom, cod, _scan=_scan)

    @property
    def params(self) -> Ty:
        """The parameter space (P) of the parametric morphism."""
        return getattr(self, "_params", Ty())

    @property
    def data_dom(self) -> Ty:
        """The data input (A), distinct from the structural dom (A @ P)."""
        return getattr(self, "_data_dom", self.dom)

    @property
    def data_cod(self) -> Ty:
        """The data output (B)."""
        return getattr(self, "_data_cod", self.cod)

    def draw(self, path=None, **params):
        """
        Draw the structural diagram.

        Parameters:
            path (str, optional) : The path where to save the drawing.
            params : Passed to :meth:`category.ar.draw`.
        """
        return super().draw(path=path, **params)

    def reparameterize(self, reparam_box: symmetric.Diagram) -> Diagram:
        """
        Reparameterize a Para morphism by a morphism on parameter spaces.

        Parameters:
            reparam_box (category.ar) : The morphism on parameter spaces.

        The reparam_box must satisfy :code:`reparam_box.cod == self.params`.
        """
        reparam_box = _as_monoidal_diagram(reparam_box)
        if reparam_box.cod != self.params:
            raise ValueError(
                f"Parameter mismatch: {reparam_box.cod} != {self.params}")
        base_self = _as_monoidal_diagram(self)
        left = symmetric.Diagram.id(self.data_dom) @ reparam_box
        target = left >> base_self
        result = Diagram(target.inside, target.dom, target.cod, _scan=False)
        result._params = reparam_box.dom
        result._data_dom = self.data_dom
        result._data_cod = self.data_cod
        return result

    reparam = reparameterize

    @classmethod
    def id(cls, dom: Ty = Ty()) -> Diagram:
        """
        The identity morphism in Para.

        Parameters:
            dom (category.ob) : The domain of the identity.
        """
        underlying = symmetric.Diagram.id(dom)
        result = Diagram(
            underlying.inside, underlying.dom, underlying.cod, _scan=False)
        result._params = Ty()
        result._data_dom = dom
        result._data_cod = dom
        return result

    def then(self, other: Diagram) -> Diagram:
        """
        Sequential composition of Para morphisms.

        Parameters:
            other (Diagram) : The other Para morphism.
        """
        other = _as_para_diagram(other)

        if self.data_cod != other.data_dom:
            raise ValueError(
                f"Data domain mismatch: {self.data_cod} != {other.data_dom}")

        base_self = _as_monoidal_diagram(self)
        base_other = _as_monoidal_diagram(other)
        base_id_Q = symmetric.Diagram.id(other.params)
        underlying_seq = (base_self @ base_id_Q) >> base_other

        result = Diagram(
            underlying_seq.inside, underlying_seq.dom, underlying_seq.cod,
            _scan=False)
        result._params = self.params @ other.params
        result._data_dom = self.data_dom
        result._data_cod = other.data_cod

        return result
    
    def tensor(self, other: Diagram) -> Diagram:
        """
        Parallel composition of Para morphisms.

        Parameters:
            other (Diagram) : The other Para morphism.
        """
        other = _as_para_diagram(other)
        base_self = _as_monoidal_diagram(self)
        base_other = _as_monoidal_diagram(other)

        # We need to swap the first params (P) with the second data input (C)
        # to keep all params at the end: A @ C @ P @ Q
        a, p = self.data_dom, self.params
        c, q = other.data_dom, other.params

        swaps = symmetric.Diagram.id(a)\
            @ symmetric.Diagram.swap(c, p)\
            @ symmetric.Diagram.id(q)

        underlying = swaps >> (base_self @ base_other)

        result = Diagram(
            underlying.inside, underlying.dom, underlying.cod, _scan=False)
        result._params = p @ q
        result._data_dom = a @ c
        result._data_cod = self.data_cod @ other.data_cod
        return result


class Box(symmetric.Box, Diagram):
    """
    A Para box is a diagram with a single box in the base category.

    Parameters:
        name (str) : The name of the box.
        dom (category.ob) : The data domain of the box.
        cod (category.ob) : The data codomain of the box.
        params (category.ob, optional) : The parameter space of the box.

    .. admonition:: Summary

        .. autosummary::

class Reparam:
    """
    A reparameterization 2-cell.

    Parameters:
        source (Diagram) : The source Para morphism.
        target (Diagram) : The target Para morphism.
        reparam_box (category.ar) : The morphism on parameter spaces.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            draw
    """
    def __init__(self, source: Diagram, target: Diagram, reparam_box, validate=True):
        self.source = _as_para_diagram(source)
        self.target = _as_para_diagram(target)
        self.reparam_box = _as_monoidal_diagram(reparam_box)
        if validate:
            self._validate()

    def _validate(self):
        if self.source.data_dom != self.target.data_dom:
            raise ValueError(
                f"Data domain mismatch: {self.source.data_dom} "
                f"!= {self.target.data_dom}")
        if self.source.data_cod != self.target.data_cod:
            raise ValueError(
                f"Data codomain mismatch: {self.source.data_cod} "
                f"!= {self.target.data_cod}")
        if self.reparam_box.cod != self.source.params:
            raise ValueError(
                f"Source parameter mismatch: {self.reparam_box.cod} "
                f"!= {self.source.params}")
        if self.reparam_box.dom != self.target.params:
            raise ValueError(
                f"Target parameter mismatch: {self.reparam_box.dom} "
                f"!= {self.target.params}")

        expected = self.source.reparameterize(self.reparam_box)
        if expected != self.target:
            raise ValueError(
                "The reparameterization square does not commute.")

    @property
    def data_dom(self):
        """ The data domain of the 2-cell. """
        return self.source.data_dom

    @property
    def data_cod(self):
        """ The data codomain of the 2-cell. """
        return self.source.data_cod

    def draw(self, path=None, **params):
        """
        Draw the reparameterization square as an equation.

        Parameters:
            path (str, optional) : The path where to save the drawing.
            params : Passed to :meth:`discopy.drawing.Equation.draw`.
        """
        from discopy.drawing import Equation
        return Equation(self.source, self.target).draw(path=path, **params)


    def then(self, other: Reparam) -> Reparam:
        """
        Vertical composition of 2-cells.

        Parameters:
            other (Reparam) : The other 2-cell.
        """
        assert_isinstance(other, Reparam)
        if self.target != other.source:
            raise ValueError("2-cells are not vertically composable.")
        reparam_box = other.reparam_box >> self.reparam_box
        return Reparam(self.source, other.target, reparam_box, validate=False)

    __rshift__ = then

    def tensor(self, other: Reparam) -> Reparam:
        """
        Horizontal composition of 2-cells.

        Parameters:
            other (Reparam) : The other 2-cell.
        """
        assert_isinstance(other, Reparam)
        reparam_box = self.reparam_box @ other.reparam_box
        source = self.source @ other.source
        target = self.target @ other.target
        return Reparam(source, target, reparam_box, validate=False)

    __matmul__ = tensor

    def __repr__(self):
        return f"{factory_name(type(self))}({self.source!r}, " \
               f"{self.target!r}, {self.reparam_box!r})"

    def __str__(self):
        return f"{self.source} => {self.target} by {self.reparam_box}"
