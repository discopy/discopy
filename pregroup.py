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
from discopy.pivotal import Ty, Box, Id, Cup, AxiomError
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
            raise ValueError("Expected str, got {} of type {} instead."
                             .format(repr(w), type(w)))
        if not isinstance(t, Ty):
            raise ValueError("Input of type Ty expected, got {} "
                             "of type {} instead.".format(repr(t), type(t)))
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
    >>> Alice, jokes = Word('Alice', n), Word('jokes', n.l @ s)
    >>> F = Model({s: 1, n: 2}, {Alice: [0, 1], jokes: [1, 1]})
    >>> assert F(Alice @ jokes >> Cup(n, n.l) @ Id(s))
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
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> assert eager_parse(Alice, loves, Bob) == grammar << Alice @ loves @ Bob
    >>> who = Word('who', n.r @ n @ s.l @ n)
    >>> eager_parse(Bob, who, loves, Alice, target=n).offsets
    [0, 1, 5, 8, 0, 2, 1, 1]
    """
    result = fold(lambda x, y: x @ y, words)
    scan = result.cod
    while True:
        fail = True
        for i in range(len(scan) - 1):
            try:
                if scan[i: i + 1].r != scan[i + 1: i + 2]:
                    raise AxiomError
                cup = Cup(scan[i: i + 1], scan[i + 1: i + 2])
                result = result >> Id(scan[: i]) @ cup @ Id(scan[i + 2:])
                scan, fail = result.cod, False
                break
            except AxiomError:
                pass
        if result.cod == target:
            return result
        if fail:
            raise FAIL


class FAIL(Exception):
    """
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> eager_parse(Alice, Bob, loves)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    discopy.pregroup.FAIL
    >>> who = Word('who', n.r @ n @ s.l @ n)
    >>> eager_parse(Alice, loves, Bob, who, loves, Alice)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    discopy.pregroup.FAIL
    """


def brute_force(*vocab, target=Ty('s')):
    """
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> gen = brute_force(Alice, loves, Bob)
    >>> assert next(gen) == Alice @ loves @ Alice >> grammar
    >>> assert next(gen) == Alice @ loves @ Bob >> grammar
    >>> assert next(gen) == Bob @ loves @ Alice >> grammar
    >>> assert next(gen) == Bob @ loves @ Bob >> grammar
    >>> gen = brute_force(Alice, loves, Bob, target=n)
    >>> next(gen)
    Word('Alice', Ty('n'))
    >>> next(gen)
    Word('Bob', Ty('n'))
    """
    test = [()]
    for words in test:
        for word in vocab:
            try:
                yield eager_parse(*(words + (word, )), target=target)
            except FAIL:
                pass
            test.append(words + (word, ))
