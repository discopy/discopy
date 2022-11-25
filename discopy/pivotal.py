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

>>> x, y = map(Ty, "xy")
>>> assert x.r == x.l and x.l.l == x == x.r.r
>>> f = Box('f', x, y)
>>> assert f.r == f.l and f.l.l == f == f.r.r

>>> from discopy import drawing
>>> drawing.equation(f.transpose(left=True), f.transpose(left=False),
...                  path="docs/imgs/pivotal/axiom.png")

.. image:: /imgs/pivotal/axiom.png
    :align: center
"""

from __future__ import annotations

from discopy import cat, monoidal, rigid
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
class Diagram(rigid.Diagram):
    """
    A pivotal diagram is a rigid diagram
    with pivotal types as domain and codomain.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    def dagger(self):
        """
        The dagger of a pivotal diagram is its vertical reflection.

        Example
        -------
        >>> from discopy import drawing
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()
        >>> drawing.equation(f, f.dagger(), symbol="$\\mapsto$",
        ...     path="docs/imgs/pivotal/dagger.png", asymmetry=.1)

        .. image:: imgs/pivotal/dagger.png
            :align: center
        """
        return cat.Arrow.dagger(self)

    def conjugate(self):
        """
        The conjugate of a diagram is the dagger of its rotation.

        Equivalently, it is the rotation of the dagger.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x @ y, z).curry()
        >>> assert f[::-1].rotate() == f.rotate()[::-1]

        >>> from discopy import drawing
        >>> drawing.equation(f, f.conjugate(), symbol="$\\mapsto$",\\
        ...     path="docs/imgs/pivotal/conjugate.png", asymmetry=.1)

        .. image:: imgs/pivotal/conjugate.png
            :align: center
        """
        return self.rotate().dagger()


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

    def drawing(self):
        return monoidal.Box.drawing(self)

    @property
    def is_conjugate(self):
        """ Whether the box is a conjugate, i.e. the transpose of a dagger. """
        return self.is_dagger and bool(self.z)


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

    def rotate(self, left=False):
        del left
        return self.cap_factory(self.right.r, self.left.r)


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

    def rotate(self, left=False):
        del left
        return self.cup_factory(self.right.r, self.left.r)


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


for cls in [Diagram, Box, Cup, Cap]:
    cls.ty_factory = Ty

Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Id = Diagram.id
