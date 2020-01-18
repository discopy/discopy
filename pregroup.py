# -*- coding: utf-8 -*-

"""
Implements disco models in the category of matrices and circuits.

>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Model(ob, ar)
>>> assert F(sentence) == True
"""

from functools import reduce as fold
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from discopy import messages
from discopy.rigidcat import Ty, Box, Diagram, Id, Cup
from discopy.matrix import MatrixFunctor
from discopy.circuit import CircuitFunctor


class Word(Box):
    """ Implements words as boxes with a pregroup type as codomain.

    >>> Alice = Word('Alice', Ty('n'))
    >>> loves = Word('loves',
    ...     Ty('n').r @ Ty('s') @ Ty('n').l)
    >>> Alice
    Word('Alice', Ty('n'))
    >>> loves
    Word('loves', Ty(Ob('n', z=1), 's', Ob('n', z=-1)))
    """
    def __init__(self, w, t, _dagger=False):
        """
        >>> Word('Alice', Ty('n'))
        Word('Alice', Ty('n'))
        """
        if not isinstance(w, str):
            raise TypeError(messages.type_err(str, w))
        if not isinstance(t, Ty):
            raise TypeError(messages.type_err(Ty, t))
        self._word, self._type = w, t
        dom, cod = (t, Ty()) if _dagger else (Ty(), t)
        Box.__init__(self, (w, t), dom, cod, _dagger=_dagger)

    def dagger(self):
        """
        >>> Word('Alice', Ty('n')).dagger()
        Word('Alice', Ty('n')).dagger()
        """
        return Word(self._word, self._type, not self._dagger)

    @property
    def word(self):
        """
        >>> Word('Alice', Ty('n')).word
        'Alice'
        """
        return self._word

    @property
    def type(self):
        """
        >>> Word('Alice', Ty('n')).type
        Ty('n')
        """
        return self._type

    def __repr__(self):
        """
        >>> Word('Alice', Ty('n'))
        Word('Alice', Ty('n'))
        >>> Word('Alice', Ty('n')).dagger()
        Word('Alice', Ty('n')).dagger()
        """
        return "Word({}, {}){}".format(repr(self.word), repr(self.type),
                                       ".dagger()" if self._dagger else "")

    def __str__(self):
        """
        >>> print(Word('Alice', Ty('n')))
        Alice
        """
        return str(self.word)


class Model(MatrixFunctor):
    """ Implements functors from pregroup grammars to matrices.

    >>> n, s = Ty('n'), Ty('s')
    >>> Alice, jokes = Word('Alice', n), Word('jokes', n.r @ s)
    >>> F = Model({s: 1, n: 2}, {Alice: [0, 1], jokes: [1, 1]})
    >>> assert F(Alice @ jokes >> Cup(n, n.r) @ Id(s))
    """
    def __repr__(self):
        """
        >>> Model({}, {Word('Alice', Ty('n')): [0, 1]})
        Model(ob={}, ar={Word('Alice', Ty('n')): [0, 1]})
        """
        return super().__repr__().replace("MatrixFunctor", "Model")


class CircuitModel(CircuitFunctor):
    """
    >>> from discopy.circuit import sqrt, H, X, Ket, CX
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> sentence = grammar << Alice @ loves @ Bob
    >>> ob = {s: 0, n: 1}
    >>> ar = {Alice: Ket(0),
    ...       loves: CX << sqrt(2) @ H @ X << Ket(0, 0),
    ...       Bob: Ket(1)}
    >>> F = CircuitModel(ob, ar)
    >>> BornRule = lambda c: abs(c.eval().array) ** 2
    >>> assert BornRule(F(sentence))
    """
    def __repr__(self):
        """
        >>> from discopy.circuit import Ket
        >>> CircuitModel({Ty('n'): 1}, {Word('Alice', Ty('n')): Ket(0)})
        CircuitModel(ob={Ty('n'): 1}, ar={Word('Alice', Ty('n')): Ket(0)})
        """
        return super().__repr__().replace("CircuitFunctor", "CircuitModel")


def eager_parse(*words, target=Ty('s')):
    """
    Tries to parse a given list of words in an eager fashion.
    """
    result = fold(lambda x, y: x @ y, words)
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
    Draws a pregroup diagram.

    Parameters
    ----------
    width : float, optional
        Width of the word triangles, default is :code:`2.0`.
    space : float, optional
        Space between word triangles, default is :code:`0.5`.
    textpad : float, optional
        Padding between text and wires, default is :code:`0.1`.
    draw_types : bool, optional
        Whether to draw type labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['equal', 'auto']`.
    fontsize : int, optional
        Font size for the words, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    margins : tuple, optional
        Margins.
    show : bool, optional
        Whether to call plt.show(), default is :code:`True`.
    """
    textpad = params.get('textpad', .1)
    space = params.get('space', .5)
    width = params.get('width', 2.)
    draw_types = params.get('draw_types', True)
    aspect = params.get('aspect', 'equal')
    margins = params.get('margins', (0, 0))
    fontsize = params.get('fontsize', 12)
    fontsize_types = params.get('fontsize_types', fontsize)
    figsize = params.get('figsize', None)
    show = params.get('show', True)

    def draw_triangles(axis, words):
        scan = []
        for i, word in enumerate(words.boxes):
            for j, _ in enumerate(word.cod):
                x_wire = (space + width) * i\
                    + (width / (len(word.cod) + 1)) * (j + 1)
                scan.append(x_wire)
                if draw_types:
                    axis.text(x_wire + textpad, -2 * textpad, str(word.cod[j]),
                              fontsize=fontsize_types)
            path = Path(
                [((space + width) * i, 0),
                 ((space + width) * i + width, 0),
                 ((space + width) * i + width / 2, 1),
                 ((space + width) * i, 0)],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
            axis.add_patch(PathPatch(path, facecolor='none'))
            axis.text((space + width) * i + width / 2, textpad,
                      str(word), ha='center', fontsize=fontsize)
        return scan

    def draw_cups_and_wires(axis, cups, scan):
        for j, off in [(j, off)
                       for j, s in enumerate(cups) for off in s.offsets]:
            middle = (scan[off] + scan[off + 1]) / 2
            verts = [(scan[off], 0),
                     (scan[off], - j - 1),
                     (middle, - j - 1)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            axis.add_patch(PathPatch(Path(verts, codes), facecolor='none'))
            verts = [(middle, - j - 1),
                     (scan[off + 1], - j - 1),
                     (scan[off + 1], 0)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            axis.add_patch(PathPatch(Path(verts, codes), facecolor='none'))
            scan = scan[:off] + scan[off + 2:]
        for i, _ in enumerate(diagram.cod):
            verts = [(scan[i], 0), (scan[i], - len(cups) - 1)]
            codes = [Path.MOVETO, Path.LINETO]
            axis.add_patch(PathPatch(Path(verts, codes)))
            if draw_types:
                axis.text(
                    scan[i] + textpad, - len(cups) - space,
                    str(diagram.cod[i]), fontsize=fontsize_types)
    if not isinstance(diagram, Diagram):
        raise TypeError(messages.type_err(Diagram, diagram))
    words, *cups = diagram.slice().boxes
    is_pregroup = all(isinstance(box, Word) for box in words.boxes)\
        and all(isinstance(box, Cup) for s in cups for box in s.boxes)
    if not is_pregroup:
        raise ValueError(messages.expected_pregroup())
    _, axis = plt.subplots(figsize=figsize)
    scan = draw_triangles(axis, words.normal_form())
    draw_cups_and_wires(axis, cups, scan)
    if margins is not None:
        plt.margins(*margins)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis('off')
    axis.set_xlim(0, (space + width) * len(words.boxes) - space)
    axis.set_ylim(- len(cups) - space, 1)
    axis.set_aspect(aspect)
    if show:  # pragma: no cover
        plt.show()
