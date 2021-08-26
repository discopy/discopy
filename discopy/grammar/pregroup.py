# -*- coding: utf-8 -*-

"""
Implements pregroup grammars and distributional compositional models.

>>> from discopy.tensor import Functor
>>> from discopy.rigid import Ty
>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Functor(ob, ar)
>>> assert F(sentence) == True

>>> draw(sentence, figsize=(4, 2),\\
...      path='docs/_static/imgs/grammar/pregroup-example.png')

.. image:: ../_static/imgs/grammar/pregroup-example.png
    :align: center
"""

from discopy import messages, drawing, rewriting, monoidal
from discopy.grammar import cfg
from discopy.rigid import Ty, Box, Diagram, Id, Cup, Cap, Swap


class Word(cfg.Word, Box):
    """ Word with a :class:`discopy.rigid.Ty` as codomain. """

    def __repr__(self):
        extra = ", dom={}".format(repr(self.dom)) if self.dom else ""
        extra += ", _dagger=True" if self._dagger else ""
        extra += ", _z={}".format(self._z) if self._z != 0 else ""
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


def normal_form(diagram, normalizer=None, **params):
    """
    Applies normal form to a pregroup diagram of the form
    `word @ ... @ word >> cups` by normalising the words and the sentences
    seperately before combining them, so it can be drawn using `grammar.draw`.
    """
    words, is_pregroup = Id(Ty()), True
    for _, box, right in diagram.layers:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box
        else:
            break

    wires = diagram[len(words):]
    is_pregroup = is_pregroup and all(
        isinstance(box, Cup) or isinstance(box, Swap) or isinstance(box, Cap)
        for box in wires.boxes)
    if not is_pregroup:
        raise ValueError(messages.expected_pregroup())

    norm = lambda d: monoidal.Diagram.normal_form(
        d, normalizer=normalizer or Diagram.normalize, **params)

    return norm(words) >> norm(wires)


normalize = rewriting.snake_removal


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
    from discopy.rigid import Swap
    if not isinstance(diagram, Diagram):
        raise TypeError(messages.type_err(Diagram, diagram))
    words, is_pregroup = Id(Ty()), True
    for _, box, right in diagram.layers:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box
        else:
            break
    cups = diagram[len(words):].foliation().boxes\
        if len(words) < len(diagram) else []
    is_pregroup = is_pregroup and words and all(
        isinstance(box, Cup) or isinstance(box, Swap)
        for s in cups for box in s.boxes)
    if not is_pregroup:
        raise ValueError(messages.expected_pregroup())
    drawing.pregroup_draw(words, cups, **params)
