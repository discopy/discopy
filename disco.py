# -*- coding: utf-8 -*-

"""
Implements distributional compositional models.

>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Model(ob, ar)
>>> assert F(sentence) == True
"""

from functools import reduce as fold
from discopy.pregroup import (
    Ob, Ty, Diagram, Box, Wire, Cup, Cap, AxiomError)
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import PRO, Id, Gate, Bra, Ket, CX, CircuitFunctor


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
    >>> assert F(Alice @ jokes >> Cup(n, n.l) @ Wire(s))
    """
    def __init__(self, ob, ar):
        """
        >>> Model({'n': 1}, {})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected a basic type, got 'n' instead.
        >>> Model({Ty('n'): 2}, {})
        Model(ob={Ty('n'): Dim(2)}, ar={})
        """
        for x in ob.keys():
            if not isinstance(x, Ty) or not x.is_basic:
                raise ValueError(
                    "Expected a basic type, got {} instead.".format(repr(x)))
        super().__init__(ob, ar, ob_cls=Ob, ar_cls=Diagram)

    def __repr__(self):
        """
        >>> Model({}, {Word('Alice', Ty('n')): [0, 1]})
        Model(ob={}, ar={Word('Alice', Ty('n')): [0, 1]})
        """
        return super().__repr__().replace("MatrixFunctor", "Model")

    def __call__(self, diagram):
        """
        >>> n, s = Ty('n'), Ty('s')
        >>> Alice, jokes = Word('Alice', n), Word('jokes', n.l @ s)
        >>> F = Model({s: 1, n: 2}, {Alice: [0, 1], jokes: [1, 1]})
        >>> F(n @ s.l)
        Dim(2)
        >>> F(Cup(n, n.l))
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[1.0, 0.0, 0.0, 1.0])
        >>> F(Cap(n, n.r))
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[1.0, 0.0, 0.0, 1.0])
        >>> F(Alice)
        Matrix(dom=Dim(1), cod=Dim(2), array=[0, 1])
        >>> F(Alice @ jokes)
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[0, 0, 1, 1])
        >>> F(Alice @ jokes >> Cup(n, n.l) @ Wire(s))
        Matrix(dom=Dim(1), cod=Dim(1), array=[1.0])
        """
        if isinstance(diagram, Ty):
            return sum([self.ob[Ty(x.name)] for x in diagram], Dim(1))
        if isinstance(diagram, Cup):
            return Matrix(self(diagram.dom), Dim(),
                          Matrix.id(self(diagram.dom[:1])).array)
        if isinstance(diagram, Cap):
            return Matrix(Dim(), self(diagram.cod),
                          Matrix.id(self(diagram.cod[:1])).array)
        if isinstance(diagram, Diagram):
            return super().__call__(diagram)
        raise ValueError(
            "Expected input of type Ty or Diagram, got {} of type {} instead"
            .format(repr(diagram), type(diagram)))


class CircuitModel(CircuitFunctor):
    """
    >>> from discopy.circuit import sqrt, H, X
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
    >>> sentence = grammar << Alice @ loves @ Bob
    >>> ob = {s: 0, n: 1}
    >>> ar = {Alice: Ket(0),
    ...       loves: CX << sqrt(2) @ H @ X << Ket(0, 0),
    ...       Bob: Ket(1)}
    >>> F = CircuitModel(ob, ar)
    >>> BornRule = lambda c: abs(c.eval().array) ** 2
    >>> assert BornRule(F(sentence))
    """
    def __call__(self, diagram):
        """
        >>> F = CircuitModel({}, {})
        >>> F('x')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected input of type Ty or Diagram, got 'x'...
        """
        unnormalised_H = Gate('H @ sqrt(2)', 1, [1, 1, 1, -1])
        if isinstance(diagram, Ty):
            return sum([self.ob[Ty(b.name)] for b in diagram], PRO(0))
        if isinstance(diagram, Cup):
            result, n = Id(self(diagram.dom)), len(self(diagram.dom)) // 2
            cup = CX >> unnormalised_H @ Id(1) >> Bra(0, 0)
            for i in range(n):
                result = result >> Id(n - i - 1) @ cup @ Id(n - i - 1)
            return result
        if isinstance(diagram, Cap):
            result, n = Id(self(diagram.cod)), len(self(diagram.cod)) // 2
            cap = CX << unnormalised_H @ Id(1) << Ket(0, 0)
            for i in range(n):
                result = result << Id(n - i - 1) @ cap @ Id(n - i - 1)
            return result
        if isinstance(diagram, Diagram):
            return super().__call__(diagram)
        raise ValueError(
            "Expected input of type Ty or Diagram, got {} of type {} instead."
            .format(repr(diagram), type(diagram)))


def eager_parse(*words, target=Ty('s')):
    """
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
    >>> assert eager_parse(Alice, loves, Bob) == grammar << Alice @ loves @ Bob
    >>> who = Word('who', n.r @ n @ s.l @ n)
    >>> eager_parse(Bob, who, loves, Alice, target=n).offsets
    [0, 1, 5, 8, 0, 2, 1, 1]
    """
    result = fold(lambda x, y: x @ y, words)
    t = result.cod
    while True:
        b = True
        for i in range(len(t) - 1):
            try:
                if t[i: i + 1].r != t[i + 1: i + 2]:
                    raise AxiomError
                cup = Cup(t[i: i + 1], t[i + 1: i + 2])
                result = result >> Wire(t[: i]) @ cup @ Wire(t[i + 2:])
                t, b = result.cod, False
                break
            except AxiomError:
                pass
        if result.cod == target:
            return result
        if b:
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
    disco.FAIL
    >>> who = Word('who', n.r @ n @ s.l @ n)
    >>> eager_parse(Alice, loves, Bob, who, loves, Alice)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    disco.FAIL
    """


def brute_force(*vocab, target=Ty('s')):
    """
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)
    >>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
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
        for w in vocab:
            try:
                yield eager_parse(*(words + (w, )), target=target)
            except FAIL:
                pass
            test.append(words + (w, ))
