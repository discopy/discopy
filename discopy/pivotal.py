# -*- coding: utf-8 -*-

"""
The free pivotal category,
i.e. diagrams with cups and caps that can rotate by a full turn.

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
    Category
    Functor

Axioms
------
A pivotal category is a rigid category where left and right transpose coincide.

>>> x, y, z = map(Ty, "xyz")
>>> assert x.r == x.l and x.l.l == x == x.r.r
>>> f = Box('f', x, y)

>>> from discopy.drawing import Equation
>>> Equation(f.transpose(left=True), f.r, f.transpose(left=False)).draw(
...     figsize=(6, 3), path="docs/_static/pivotal/axiom.png")

.. image:: /_static/pivotal/axiom.png
    :align: center

For each diagram, we have its conjugate:

>>> d = Box('g', x @ y, z).curry()
>>> Equation(d, d.conjugate(), symbol="").draw(
...     figsize=(6, 2), space=2, path="docs/_static/pivotal/box-conjugate.png")

.. image:: /_static/pivotal/box-conjugate.png
    :align: center

We also have its dagger and its transpose:

>>> Equation(d.dagger(), d.rotate(), symbol="").draw(
...     figsize=(6, 2), space=2,
...     path="docs/_static/pivotal/dagger-transpose.png")

.. image:: /_static/pivotal/dagger-transpose.png
    :align: center
"""

from __future__ import annotations

from discopy import cat, rigid, traced
from discopy.cat import factory


class Ob(rigid.Ob):
    """
    A pivotal object is a rigid object where left and right adjoints coincide.

    Parameters:
        name : The name of the object.
        z (bool) : Whether the object is an adjoint or not.
    """
    l = r = property(lambda self: type(self)(self.name, (self.z + 1) % 2))


@factory
class Ty(rigid.Ty):
    """
    A pivotal type is a rigid type with pivotal objects inside.

    Parameters:
        inside (Ob) : The objects inside the type.
    """
    ob_factory = Ob


@factory
class PRO(rigid.PRO, Ty):
    """
    A pivotal PRO is a natural number ``n``
    seen as a pivotal type of length ``n``.

    Parameters
    ----------
    n : int
        The length of the PRO type.
    """
    __ambiguous_inheritance__ = (rigid.PRO, )
    l = r = property(lambda self: self)


@factory
class Diagram(rigid.Diagram, traced.Diagram):
    """
    A pivotal diagram is a rigid diagram and a traced diagram
    with pivotal types as domain and codomain.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ty_factory = Ty

    def dagger(self):
        """
        The dagger of a pivotal diagram is its vertical reflection.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()

        >>> from discopy.drawing import Equation
        >>> Equation(f, f.dagger(), symbol="$\\\\mapsto$").draw(
        ...     figsize=(6, 3), asymmetry=.1,
        ...     path="docs/_static/pivotal/dagger.png")

        .. image:: /_static/pivotal/dagger.png
            :align: center
        """
        return cat.Arrow.dagger(self)

    def conjugate(self):
        """
        The horizontal reflection of a diagram,
        defined as the dagger of the rotation.

        Equivalently, it is the rotation of the dagger.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()
        >>> assert f.conjugate() == f[::-1].rotate() == f.rotate()[::-1]

        >>> from discopy.drawing import Equation
        >>> Equation(f, f.conjugate(), symbol="$\\\\mapsto$").draw(
        ...     figsize=(6, 3), path="docs/_static/pivotal/conjugate.png")

        .. image:: /_static/pivotal/conjugate.png
            :align: center
        """
        return self.rotate().dagger()

    @classmethod
    def trace_factory(cls, diagram: Diagram, left=False):
        """
        The trace of a pivotal diagram is its pre- and post-composition with
        cups and caps to form a feedback loop.

        Parameters:
            diagram : The diagram to trace.
            left : Whether to trace on the left or right.
        """
        traced_wire = diagram.dom[:1] if left else diagram.dom[-1:]
        dom, cod = (diagram.dom[1:], diagram.cod[1:]) if left\
            else (diagram.dom[:-1], diagram.cod[:-1])
        return cls.cap_factory(traced_wire.r, traced_wire) @ dom\
            >> traced_wire.r @ diagram\
            >> cls.cup_factory(traced_wire.r, traced_wire) @ cod if left\
            else dom @ cls.cap_factory(traced_wire, traced_wire.r)\
            >> diagram @ traced_wire.r\
            >> cod @ cls.cup_factory(traced_wire, traced_wire.r)


class Box(rigid.Box, Diagram):
    """
    A pivotal box is a rigid box in a pivotal diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (rigid.Box, )

    def rotate(self, left=False):
        del left
        return type(self)(
            self.name, dom=self.cod.r, cod=self.dom.r,
            data=self.data, is_dagger=self.is_dagger, z=(self.z + 1) % 2)

    def dagger(self) -> Box:
        return type(self)(
            name=self.name, dom=self.cod, cod=self.dom,
            data=self.data, is_dagger=not self.is_dagger, z=self.z)

    @property
    def is_conjugate(self):
        """ Whether the box is a conjugate, i.e. the transpose of a dagger. """
        return self.is_dagger and bool(self.z)

    def to_drawing(self):
        result = super().to_drawing()
        result.is_conjugate = self.is_conjugate
        return result


class Cup(rigid.Cup, Box):
    """
    A pivotal cup is a rigid cup of pivotal types.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (rigid.Cup, )

    def dagger(self) -> Cap:
        """ The dagger of a pivotal cup. """
        return self.cap_factory(self.left, self.right)


class Cap(rigid.Cap, Box):
    """
    A pivotal cap is a rigid cap of pivotal types.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (rigid.Cap, )

    def dagger(self) -> Cup:
        """ The dagger of a pivotal cap. """
        return self.cup_factory(self.left, self.right)


class Category(rigid.Category):
    """
    A pivotal category is a rigid category
    where left and right adjoints coincide.

    Parameters:
    ob : The type of objects.
    ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(rigid.Functor):
    """
    A pivotal functor is a rigid functor on a pivotal category.

    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)


Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Id = Diagram.id
