# -*- coding: utf-8 -*-

"""
A pregroup grammar is a free rigid category with words as boxes.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Cup
    Cap
    Swap
    Word
    Category
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        eager_parse
        brute_force
"""

from discopy import rigid, frobenius, messages
from discopy.cat import factory
from discopy.utils import AxiomError
from discopy.grammar import thue
from discopy.rigid import Ob  # noqa: F401


@factory
class Ty(rigid.Ty):
    """
    A pregroup type is a rigid type.

    Parameters:
        inside (tuple[Ob, ...]) : The objects inside the type.

    Note
    ----
    In order to define more general DisCoCat diagrams, pregroup types do not
    raise any AxiomError if you build cups and caps with the wrong adjoints.

    Example
    -------
    >>> n = Ty('n')
    >>> n.assert_isadjoint(n.l)
    >>> n.assert_isadjoint(n.r)
    """
    def assert_isadjoint(self, other):
        """
        Raise ``AxiomError`` if two pregroup types are not adjoints.

        Parameters:
            other : The alleged right adjoint.
        """
        if self.r != other and self != other.r:
            raise AxiomError(messages.NOT_ADJOINT.format(self, other))


@factory
class Diagram(frobenius.Diagram):
    """
    A pregroup diagram is a rigid diagram with :class:`Word` boxes.

    Parameters:
        inside (tuple[frobenius.Layer, ...]) : The layers of the diagram.
        dom (rigid.Ty) : The domain of the diagram, i.e. its input.
        cod (rigid.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    In order to define more general DisCoCat diagrams, pregroup diagrams
    subclass frobenius rather than rigid. Have fun with swaps and spiders!

    Example
    -------
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Word('Alice', n), Word('Bob', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> sentence = grammar << Alice @ loves @ Bob
    >>> print(sentence[:4])
    Alice >> n @ loves >> n @ n.r @ s @ n.l @ Bob >> Cup(n, n.r) @ s @ n.l @ n
    >>> from discopy import tensor
    >>> ob = {s: 1, n: 2}
    >>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
    >>> F = tensor.Functor(ob, ar, dom=Category(), dtype=bool)
    >>> assert F(sentence)
    """
    ty_factory = Ty

    def normal_form(self, **params):
        """
        Applies normal form to a pregroup diagram of the form
        ``word @ ... @ word >> wires`` by normalising words and wires
        seperately before combining them, so it can be drawn with :meth:`draw`.
        """
        words, is_pregroup = self.id(), True
        for _, box, right in self.inside:
            if isinstance(box, Word):
                if right:  # word boxes should be tensored left to right.
                    is_pregroup = False
                    break
                words = words @ box
            else:
                break

        wires = self[len(words):]
        is_pregroup = is_pregroup and all(
            isinstance(box, (Cup, Cap, Swap)) for box in wires.boxes)
        if not is_pregroup or not words.cod:
            return rigid.Diagram.normal_form(self)
        return rigid.Diagram.normal_form(words)\
            >> rigid.Diagram.normal_form(wires)

    @classmethod
    def fa(cls, left, right):
        return left @ cls.cups(right.l, right)

    @classmethod
    def ba(cls, left, right):
        return cls.cups(left, left.r) @ right

    @classmethod
    def fc(cls, left, middle, right):
        return left @ cls.cups(middle.l, middle) @ right.l

    @classmethod
    def bc(cls, left, middle, right):
        return left.r @ cls.cups(middle, middle.r) @ right

    @classmethod
    def fx(cls, left, middle, right):
        return left @ cls.swap(middle.l, right.r) @ middle >>\
            cls.swap(left, right.r) @ cls.cups(middle.l, middle)

    @classmethod
    def bx(cls, left, middle, right):
        return middle @ cls.swap(left.l, middle.r) @ right >>\
            cls.cups(middle, middle.r) @ cls.swap(left.l, right)

    __eq__, __hash__ = rigid.Diagram.__eq__, rigid.Diagram.__hash__
    cups = classmethod(rigid.Diagram.cups.__func__)
    caps = classmethod(rigid.Diagram.caps.__func__)


class Box(frobenius.Box, Diagram):
    """
    A pregroup box is a frobenius box in a pregroup diagram.
    """
    rotate = rigid.Box.rotate


class Cup(frobenius.Cup, Box):
    """
    A pregroup cup is a frobenius cup in a pregroup diagram.
    """


class Cap(frobenius.Cap, Box):
    """
    A pregroup cap is a frobenius cap in a pregroup diagram.
    """


class Swap(frobenius.Swap, Box):
    """
    A pregroup swap is a frobenius swap in a pregroup diagram.
    """


class Spider(frobenius.Spider, Box):
    """
    A pregroup spider is a frobenius spider in a pregroup diagram.
    """
    def rotate(self, left=False):
        typ = self.typ.l if left else self.typ.r
        return type(self)(len(self.cod), len(self.dom), typ, self.phase)


class Word(thue.Word, Box):
    """
    A word is a rigid box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.
    """
    def __init__(self, name: str, cod: rigid.Ty, dom: rigid.Ty = Ty(),
                 **params):
        Box.__init__(self, name, dom, cod, **params)

    def __repr__(self):
        extra = f", dom={repr(self.dom)}" if self.dom else ""
        extra += ", is_dagger=True" if self.is_dagger else ""
        extra += f", z={self.z}" if self.z != 0 else ""
        return f"Word({repr(self.name)}, {repr(self.cod)}{extra})"


class Category(frobenius.Category):
    """ A pregroup category has rigid types and frobenius diagrams. """
    ob = Ty
    ar = Diagram


class Functor(frobenius.Functor):
    """ A pregroup functor is a frobenius functor with a pregroup domain. """
    dom = cod = Category()


def eager_parse(*words, target=Ty('s')):
    """
    Tries to parse a given list of words in an eager fashion.
    """
    result = Id(Ty()).tensor(*words)
    scan = result.cod
    while True:
        fail = True
        for i in range(len(scan) - 1):
            if scan[i: i + 1].r != scan[i + 1: i + 2]:
                continue
            cup = Cup(scan[i: i + 1], scan[i + 1: i + 2])
            result = result >> Id(scan[: i]) @ cup @ Id(scan[i + 2:])
            scan, fail = result.cod, False
            break
        if result.cod == target:
            return result
        if fail:
            raise NotImplementedError


def brute_force(*vocab, target=Ty('s')):
    """
    Given a vocabulary, search for grammatical sentences.
    """
    test = [()]
    for words in test:
        for word in vocab:
            try:
                yield eager_parse(*(words + (word, )), target=target)
            except NotImplementedError:
                pass
            test.append(words + (word, ))


Diagram.braid_factory, Diagram.spider_factory = Swap, Spider
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap

Id = Diagram.id
