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

from dataclasses import dataclass
import warnings
from discopy import symmetric
from discopy.drawing import Drawing, Equation
from discopy.cat import factory
from discopy.symmetric import Ty
from discopy.utils import (
    Composable, Whiskerable, NamedGeneric, assert_isinstance,
    unbiased, factory_name)


def _as_para_diagram(diagram):
    """Map a symmetric monoidal diagram into a Para diagram with empty parameters."""
    if isinstance(diagram, Diagram):
        return diagram
    return Diagram(diagram, diagram.dom, diagram.cod, diagram.ty_factory())


def _as_symmetric_diagram(diagram):
    """Remove parameters and keep only the underlying diagram."""
    if isinstance(diagram, Diagram):
        return diagram.inside
    return diagram


@factory
@dataclass
class Diagram(Composable, Whiskerable, NamedGeneric['category']):
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
    category = symmetric.Category  # default base category
    inside: category.ar
    dom: category.ob = None
    cod: category.ob = None
    params: category.ob = None

    def __post_init__(self):
        if self.dom is None:
            self.dom = self.inside.dom
        if self.cod is None:
            self.cod = self.inside.cod
        if self.params is None:
            self.params = self.inside.ty_factory()
        if self.inside.dom != self.dom + self.params:
            raise ValueError(f"Domain mismatch: {self.inside.dom} "
                             f"!= {self.dom} + {self.params}")
        if self.inside.cod != self.cod:
            raise ValueError(f"Codomain mismatch: {self.inside.cod} "
                             f"!= {self.cod}")

    @property
    def ty_factory(self):
        """ The type factory of the base category. """
        return self.dom.factory if hasattr(self.dom, "factory") else type(self.dom)

    def to_drawing(self):
        """
        Convert the Para diagram into a drawing of its structural diagram.
        """
        if hasattr(self.inside, "to_drawing"):
            return self.inside.to_drawing()
        
        return Drawing.from_box(self.inside) if hasattr(self.inside, "name")\
            else self.inside

    def draw(self, path=None, **params):
        """
        Draw the structural diagram.

        Parameters:
            path (str, optional) : The path where to save the drawing.
            params : Passed to :meth:`category.ar.draw`.
        """
        if not hasattr(self.inside, "draw"):
            warnings.warn(
                f"Base category {self.category} does not support drawing.")
            return None
        return self.inside.draw(path=path, **params)

    def reparameterize(self, reparam_box: category.ar) -> Diagram:
        """
        Reparameterize a Para morphism by a morphism on parameter spaces.

        Parameters:
            reparam_box (category.ar) : The morphism on parameter spaces.

        The reparam_box must satisfy :code:`reparam_box.cod == self.params`.
        """
        reparam_box = _as_symmetric_diagram(reparam_box)
        if reparam_box.cod != self.params:
            raise ValueError(
                f"Parameter space type mismatch: "
                f"{reparam_box.cod} != {self.params}")
        left = self.category.ar.id(self.dom) @ reparam_box
        return self.factory(
            left >> self.inside, self.dom, self.cod, reparam_box.dom)

    reparam = reparameterize

    @classmethod
    def id(cls, dom: category.ob = None) -> Diagram:
        """
        The identity morphism in Para.

        Parameters:
            dom (category.ob) : The domain of the identity.
        """
        dom = dom or cls.category.ob()
        return cls(cls.category.ar.id(dom), dom, dom, cls.category.ob())

    def then(self, other: Diagram) -> Diagram:
        """
        Sequential composition of Para morphisms.

        Parameters:
            other (Diagram) : The other Para morphism.
        """
        other = _as_para_diagram(other)
        if self.cod != other.dom:
            raise ValueError(
                f"Data domain mismatch: {self.cod} != {other.dom}")
        id_Q = self.category.ar.id(other.params)
        return self.factory((self.inside @ id_Q) >> other.inside,
                            self.dom, other.cod, self.params + other.params)

    @unbiased
    def tensor(self, other: Diagram) -> Diagram:
        """
        Parallel composition of Para morphisms.

        Parameters:
            other (Diagram) : The other Para morphism.
        """
        other = _as_para_diagram(other)
        a, p = self.dom, self.params
        c, q = other.dom, other.params
        swap = self.category.ar.swap(c, p)
        swaps = self.category.ar.id(a) @ swap @ self.category.ar.id(q)
        return self.factory(swaps >> (self.inside @ other.inside),
                            a + c, self.cod + other.cod, p + q)

    @classmethod
    def swap(cls, left: category.ob, right: category.ob) -> Diagram:
        """
        The swap morphism in Para.

        Parameters:
            left (category.ob) : The left domain.
            right (category.ob) : The right domain.
        """
        return cls(cls.category.ar.swap(left, right),
                   left + right, right + left, cls.category.ob())

    @classmethod
    def copy(cls, dom: category.ob, n: int = 2) -> Diagram:
        """
        The copy morphism in Para.

        Parameters:
            dom (category.ob) : The domain of the copy.
            n (int) : The number of copies.
        """
        return cls(cls.category.ar.copy(dom, n),
                   dom, dom ** n, cls.category.ob())


class Box(Diagram):
    """
    A Para box is a diagram with a single box in the base category.

    Parameters:
        name (str) : The name of the box.
        dom (category.ob) : The data domain of the box.
        cod (category.ob) : The data codomain of the box.
        params (category.ob, optional) : The parameter space of the box.

    .. admonition:: Summary

        .. autosummary::

            __init__
    """
    def __init__(
            self, name: str, dom: category.ob, cod: category.ob,
            params: category.ob = None, **kwargs):
        import importlib
        params = params or (dom.factory() if hasattr(dom, "factory")
                            else self.category.ob())
        try:
            module = importlib.import_module(self.category.ar.__module__)
            box_cls = getattr(module, "Box")
        except (ImportError, AttributeError):
            box_cls = symmetric.Box
        inside = box_cls(name, dom + params, cod, **kwargs)
        super().__init__(inside, dom, cod, params)


@dataclass
class Reparam(Composable, Whiskerable, NamedGeneric['category']):
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
    category = symmetric.Category

    source: Diagram
    target: Diagram
    reparam_box: category.ar

    def __post_init__(self):
        self.source = _as_para_diagram(self.source)
        self.target = _as_para_diagram(self.target)
        self.reparam_box = _as_symmetric_diagram(self.reparam_box)
        if self.source.dom != self.target.dom:
            raise ValueError(
                f"Data domain mismatch: {self.source.dom} "
                f"!= {self.target.dom}")
        if self.source.cod != self.target.cod:
            raise ValueError(
                f"Data codomain mismatch: {self.source.cod} "
                f"!= {self.target.cod}")
        if self.reparam_box.cod != self.source.params:
            raise ValueError(
                f"Source parameter mismatch: {self.reparam_box.cod} "
                f"!= {self.source.params}")
        if self.reparam_box.dom != self.target.params:
            raise ValueError(
                f"Target parameter mismatch: {self.reparam_box.dom} "
                f"!= {self.target.params}")

    @classmethod
    def id(cls, dom: Diagram) -> Reparam:
        """ The identity reparameterization 2-cell. """
        return cls(dom, dom, cls.category.ar.id(dom.params))

    @property
    def dom(self):
        """ The data domain of the 2-cell. """
        return self.source.dom

    @property
    def cod(self):
        """ The data codomain of the 2-cell. """
        return self.source.cod

    def draw(self, path=None, **params):
        """
        Draw the reparameterization square as an equation.

        Parameters:
            path (str, optional) : The path where to save the drawing.
            params : Passed to :meth:`discopy.drawing.Equation.draw`.
        """
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
        return type(self)(self.source, other.target, reparam_box)

    @unbiased
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
        return type(self)(source, target, reparam_box)

    def __repr__(self):
        return f"{factory_name(type(self))}({self.source!r}, " \
               f"{self.target!r}, {self.reparam_box!r})"

    def __str__(self):
        return f"{self.source} => {self.target} by {self.reparam_box}"


class Category(symmetric.Category):
    """
    Syntactic sugar for `Category(Ty[category.ob], Para[category])`.

    .. admonition:: Summary

        .. autosummary::

            __init__
    """
    def __init__(self, ob: type = None, ar: type = None):
        ar = Diagram if ar is None else Diagram[symmetric.Category(ob, ar)]
        ob = Ty if ob is None else Ty[ob]
        super().__init__(ob, ar)
