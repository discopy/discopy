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
        draw
"""
from __future__ import annotations

from discopy import messages, drawing, grammar, rigid, symmetric
from discopy.cat import factory
from discopy.rigid import Ty
from discopy.utils import assert_isinstance


@factory
class Diagram(rigid.Diagram):
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

        return rigid.Diagram.normal_form(words)\
            >> rigid.Diagram.normal_form(wires)


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


class Word(Box):
    """
    A word is a rigid box with a ``name``, a grammatical type as ``cod`` and
    an optional domain ``dom``.
    """
    def __init__(self, name: str, cod: rigid.Ty, dom: rigid.Ty = Ty(),
                 **params):
        super().__init__(name=name, dom=dom, cod=cod, **params)

    def __repr__(self):
        extra = ", dom={}".format(repr(self.dom)) if self.dom else ""
        extra += ", is_dagger=True" if self.is_dagger else ""
        extra += ", z={}".format(self.z) if self.z != 0 else ""
        return "Word({}, {}{})".format(
            repr(self.name), repr(self.cod), extra)


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


def draw(diagram, **params):
    """
    Draws a pregroup diagram, i.e. of shape :code:`word @ ... @ word >> cups`.

    Parameters
    ----------
    width : float, optional
        Width of the word triangles, default is :code:`2.0`.
    space : float, optional
        Space between word triangles, default is :code:`0.5`.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.2)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['equal', 'auto']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the words, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if :code:`None` we call :code:`plt.show()`.
    pretty_types : bool, optional
        Whether to draw type labels with superscript, default is :code:`False`.
    triangles : bool, optional
        Whether to draw words as triangular states, default is :code:`False`.

    Raises
    ------
    ValueError
        Whenever the input is not a pregroup diagram.
    """
    assert_isinstance(diagram, Diagram)
    words, is_pregroup = Diagram.id(), True
    for _, box, right in diagram.inside:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box.drawing()
        else:
            break
    layers = diagram[len(words):].foliation().inside\
        if len(words) < len(diagram) else ()
    is_pregroup = is_pregroup and words and all(
        isinstance(x, (Ty, Cup, Swap)) for layer in layers for x in layer)
    has_swaps = any(isinstance(x, Swap) for layer in layers for x in layer)
    if not is_pregroup:
        raise ValueError(messages.NOT_PREGROUP)
    drawing.pregroup_draw(
        words, [layer.drawing() for layer in layers], has_swaps, **params)


Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.braid_factory, Diagram.draw = Swap, draw

Id = Diagram.id
