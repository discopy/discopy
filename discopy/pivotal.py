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
"""

from __future__ import annotations

from discopy import rigid
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
        inside (tuple[Ob, ...]) : The objects inside the type.
    """
    ob_factory = Ob


@factory
class Diagram(rigid.Diagram):
    """
    A pivotal diagram is a rigid diagram
    with pivotal types as domain and codomain.

    Parameters:
        inside (tuple[rigid.Layer, ...]) : The layers of the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """


class Box(rigid.Box, Diagram):
    """
    A pivotal box is a rigid box in a pivotal diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (rigid.Box, )


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
        return Cap(self.dom[0], self.dom[1])


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
        return Cup(self.cod[0], self.cod[1])


Diagram.cup_factory, Diagram.cap_factory = Cup, Cap


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
