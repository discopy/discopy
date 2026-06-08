# -*- coding: utf-8 -*-

"""
The category of parametric maps, Para.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Reparam

Axioms
------
We can check that :meth:`Diagram.tensor` follows the Para axiom:
:code:`(f @ g).dom == f.data_dom @ g.data_dom @ f.params @ g.params`

>>> a, b, c, d = map(Ty, "ABCD")
>>> p, q = map(Ty, "PQ")
>>> f = Box("f", a, b, p)
>>> g = Box("g", c, d, q)
>>> assert (f @ g).params == p @ q
>>> assert (f @ g).data_dom == a @ c
>>> assert (f @ g).dom == a @ c @ p @ q

We can check that :meth:`Diagram.then` follows the Para axiom:
:code:`(f >> g).params == f.params @ g.params`

>>> h = Box("h", b, c, p)
>>> assert (f >> h).params == p @ p
>>> assert (f >> h).data_dom == a
>>> assert (f >> h).dom == a @ p @ p
"""

from __future__ import annotations

from discopy import monoidal, symmetric
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.utils import assert_isinstance, factory_name


def _as_para_diagram(diagram):
    """Map a monoidal diagram into a Para diagram with empty parameters."""
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
    A Para diagram is a symmetric diagram with a parameter space.

    Parameters:
        inside (tuple[Layer, ...]) : The layers of the diagram.
        dom (monoidal.Ty) : The domain of the diagram.
        cod (monoidal.Ty) : The codomain of the diagram.
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
            params : Passed to :meth:`symmetric.Diagram.draw`.
        """
        return super().draw(path=path, **params)

    def reparameterize(self, reparam_box: symmetric.Diagram) -> Diagram:
        """
        Reparameterize a Para morphism by a morphism on parameter spaces.

        Parameters:
            reparam_box (symmetric.Diagram) : The morphism on parameter spaces.

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
            dom (monoidal.Ty) : The domain of the identity.
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

        Axiom: (P, f) >> (Q, g) = (P @ Q, (f @ id_Q) >> g)
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

        Axiom: (P, f) @ (Q, g) = (P @ Q, (id_A @ swap(C, P) @ id_Q) >> (f @ g))
        where f: A @ P -> B and g: C @ Q -> D.
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
    A Para box is a symmetric box with a parameter space.

    Parameters:
        name (str) : The name of the box.
        data_dom (monoidal.Ty) : The data domain of the box.
        data_cod (monoidal.Ty) : The data codomain of the box.
        params (monoidal.Ty, optional) : The parameter space of the box.
    """
    def __init__(
            self, name: str, data_dom: Ty, data_cod: Ty, params: Ty = Ty(),
            **kwargs):
        self._params = params
        self._data_dom = data_dom
        self._data_cod = data_cod

        structural_dom = data_dom @ params
        super().__init__(name, structural_dom, data_cod, **kwargs)


class Reparam:
    """
    A reparameterization 2-cell.

    Parameters:
        source (Diagram) : The source Para morphism.
        target (Diagram) : The target Para morphism.
        reparam_box (symmetric.Diagram) : The morphism on parameter spaces.
        validate (bool, optional) : Whether to validate the reparameterization.
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
