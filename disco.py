""" Implements free rigid categories and distributional compositional models.

>>> s, n = Pregroup('s'), Pregroup('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Model(ob, ar)
>>> assert F(sentence) == True
"""


from discopy.pregroup import Adjoint, Pregroup, Diagram, Box, Wire, Cup, Cap
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import (
    CircuitFunctor, Circuit, Gate, PRO, Bra, Ket)


class Word(Box):
    """ Implements words as boxes with a pregroup type as codomain.

    >>> Alice = Word('Alice', Pregroup('n'))
    >>> loves = Word('loves', Pregroup('n').r @ Pregroup('s') @ Pregroup('n').l)
    >>> Alice
    Word('Alice', Pregroup('n'))
    >>> loves
    Word('loves', Pregroup(Adjoint('n', 1), 's', Adjoint('n', -1)))
    """
    def __init__(self, w, t, _dagger=False):
        """
        >>> Word('Alice', Pregroup('n'))
        Word('Alice', Pregroup('n'))
        """
        if not isinstance(w, str):
            raise ValueError("Expected str, got {} of type {} instead."
                             .format(repr(w), type(w)))
        if not isinstance(t, Pregroup):
            raise ValueError("Input of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(t), type(t)))
        self._word, self._type = w, t
        dom, cod = (t, Pregroup()) if _dagger else (Pregroup(), t)
        Box.__init__(self, (w, t), dom, cod, _dagger=_dagger)

    def dagger(self):
        """
        >>> Word('Alice', Pregroup('n')).dagger()
        Word('Alice', Pregroup('n')).dagger()
        """
        return Word(self._word, self._type, not self._dagger)

    @property
    def word(self):
        """
        >>> Word('Alice', Pregroup('n')).word
        'Alice'
        """
        return self._word

    @property
    def type(self):
        """
        >>> Word('Alice', Pregroup('n')).type
        Pregroup('n')
        """
        return self._type

    def __repr__(self):
        """
        >>> Word('Alice', Pregroup('n'))
        Word('Alice', Pregroup('n'))
        >>> Word('Alice', Pregroup('n')).dagger()
        Word('Alice', Pregroup('n')).dagger()
        """
        return "Word({}, {}){}".format(repr(self.word), repr(self.type),
                                       ".dagger()" if self._dagger else "")

    def __str__(self):
        """
        >>> print(Word('Alice', Pregroup('n')))
        Alice
        """
        return str(self.word)

class Model(MatrixFunctor):
    """ Implements functors from pregroup grammars to matrices

    >>> n, s = Pregroup('n'), Pregroup('s')
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
        >>> Model({Pregroup('n'): 2}, {})
        Model(ob={Pregroup('n'): Dim(2)}, ar={})
        """
        for x in ob.keys():
            if not isinstance(x, Pregroup) or not x.is_basic:
                raise ValueError(
                    "Expected a basic type, got {} instead.".format(repr(x)))
        super().__init__(ob, ar)

    def __repr__(self):
        """
        >>> Model({}, {Word('Alice', Pregroup('n')): [0, 1]})
        Model(ob={}, ar={Word('Alice', Pregroup('n')): [0, 1]})
        """
        return super().__repr__().replace("MatrixFunctor", "Model")

    def __call__(self, d):
        """
        >>> n, s = Pregroup('n'), Pregroup('s')
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
        if isinstance(d, Pregroup):
            return sum([self.ob[Pregroup(x._basic)] for x in d], Dim(1))
        elif isinstance(d, Cup):
            return Matrix(self(d.dom), Dim(), Matrix.id(self(d.dom[:1])).array)
        elif isinstance(d, Cap):
            return Matrix(Dim(), self(d.cod), Matrix.id(self(d.cod[:1])).array)
        elif isinstance(d, Box):
            if d._dagger:
                return self(d.dagger()).dagger()
            return Matrix(self(d.dom), self(d.cod), self.ar[d])
        elif isinstance(d, Diagram):
            return super().__call__(d)
        else: raise ValueError("Expected Diagram")
