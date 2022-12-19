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

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        eager_parse
        brute_force
"""

from discopy import messages, rigid, symmetric
from discopy.cat import factory
from discopy.grammar import thue
from discopy.rigid import Ty


@factory
class Diagram(rigid.Diagram, symmetric.Diagram):
    """
    A pregroup diagram is a rigid diagram with :class:`Word` boxes.

    Parameters:
        inside (tuple[rigid.Layer, ...]) : The layers of the diagram.
        dom (rigid.Ty) : The domain of the diagram, i.e. its input.
        cod (rigid.Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Word('Alice', n), Word('Bob', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> sentence = grammar << Alice @ loves @ Bob
    >>> print(sentence[:4])
    Alice >> n @ loves >> n @ n.r @ s @ n.l @ Bob >> Cup(n, n.r) @ s @ n.l @ n
    >>> from discopy import tensor, rigid
    >>> ob = {s: 1, n: 2}
    >>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
    >>> F = tensor.Functor(ob, ar, dom=rigid.Category(), dtype=bool)
    >>> assert F(sentence)
    """
    def normal_form(diagram, **params):
        """
        Applies normal form to a pregroup diagram of the form
        ``word @ ... @ word >> wires`` by normalising words and wires
        seperately before combining them, so it can be drawn with :meth:`draw`.
        """
        words, is_pregroup = Id(Ty()), True
        for _, box, right in diagram.inside:
            if isinstance(box, Word):
                if right:  # word boxes should be tensored left to right.
                    is_pregroup = False
                    break
                words = words @ box
            else:
                break

        wires = diagram[len(words):]
        is_pregroup = is_pregroup and all(
            isinstance(box, (Cup, Cap, Swap)) for box in wires.boxes)
        if not is_pregroup:
            raise ValueError(messages.NOT_PREGROUP)
        if words.cod == Ty():
            return rigid.Diagram.normal_form(wires)
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


class Box(rigid.Box, Diagram):
    """
    A pregroup box is a rigid box in a pregroup diagram.
    """


class Cup(rigid.Cup, Box):
    """
    A pregroup cup is a rigid cup in a pregroup diagram.
    """


class Cap(rigid.Cap, Box):
    """
    A pregroup cap is a rigid cap in a pregroup diagram.
    """


class Swap(symmetric.Swap, Box):
    """
    A pregroup swap is a symmetric swap in a pregroup diagram.
    """
    z = 0


class Word(thue.Word, Box):
    """
    A word is a rigid box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.
    """

    z = 0

    def __init__(self, name: str, cod: rigid.Ty, dom: rigid.Ty = Ty(),
                 **params):
        super().__init__(name=name, dom=dom, cod=cod, **params)

    def __repr__(self):
        extra = f", dom={repr(self.dom)}" if self.dom else ""
        extra += ", is_dagger=True" if self.is_dagger else ""
        extra += f", z={self.z}" if self.z != 0 else ""
        return f"Word({repr(self.name)}, {repr(self.cod)}{extra})"


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


Diagram.cup_factory, Diagram.cap_factory, Diagram.braid_factory\
    = Cup, Cap, Swap

Id = Diagram.id
