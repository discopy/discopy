# -*- coding: utf-8 -*-

"""
Implements quantum circuits as diagrams and circuit-valued monoidal functors.

>>> from discopy.gates import X, Bra, Ket
>>> n = Ty('n')
>>> Alice = Box('Alice', Ty(), n)
>>> loves = Box('loves', n, n)
>>> Bob = Box('Bob', n, Ty())
>>> ob, ar = {n: 1}, {Alice: Ket(0), loves: X, Bob: Bra(1)}
>>> F = CircuitFunctor(ob, ar)
>>> F(Alice >> loves >> Bob)
Circuit(0, 0, [Ket(0), Gate('X', 1, [0, 1, 1, 0]), Bra(1)], [0, 0, 0])
>>> assert F(Alice >> loves >> Bob).eval()
"""

from discopy import config
from discopy.cat import Quiver
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor
from discopy.matrix import Dim, Matrix, MatrixFunctor

try:
    import jax.numpy as np
except ImportError:
    import numpy as np


class PRO(Ty):
    """ Implements the objects of a PRO, i.e. a non-symmetric PROP.
    Wraps a natural number n into a unary type Ty(1, ..., 1) of length n.

    >>> PRO(1) @ PRO(1)
    PRO(2)
    >>> assert PRO(3) == Ty(1, 1, 1)
    """
    def __init__(self, n):
        """
        >>> list(PRO(0))
        []
        >>> list(PRO(1))
        [Ob(1)]
        >>> assert all(len(PRO(n)) == n for n in range(5))
        """
        n = n if isinstance(n, int) else len(n)
        super().__init__(*(n * [Ob(1)]))

    def __repr__(self):
        """
        >>> PRO(0), PRO(1)
        (PRO(0), PRO(1))
        """
        return "PRO({})".format(len(self))

    def __str__(self):
        """
        >>> print(PRO(2 * 3 * 7))
        PRO(42)
        """
        return repr(self)

    def __add__(self, other):
        """
        >>> sum((PRO(n) for n in range(5)), PRO(0))
        PRO(10)
        """
        return PRO(len(self) + len(other))

    def __getitem__(self, key):
        """
        >>> PRO(42)[2:4]
        PRO(2)
        >>> assert all(PRO(42)[i] == Ob(1) for i in range(42))
        """
        if isinstance(key, slice):
            return PRO(len(super().__getitem__(key)))
        return super().__getitem__(key)


class Circuit(Diagram):
    """ Implements quantum circuits as diagrams.

    >>> from discopy.gates import CX
    >>> circuit = CX >> CX >> CX >> CX >> CX >> CX
    >>> assert np.all(circuit.eval() == Id(2).eval())
    """
    def __init__(self, dom, cod, gates, offsets):
        """
        >>> from discopy.gates import CX
        >>> c = Circuit(2, 2, [CX, CX], [0, 0])
        """
        self._gates = gates
        super().__init__(PRO(dom), PRO(cod), gates, offsets)

    @property
    def gates(self):
        """
        >>> from discopy.gates import X
        >>> Circuit(1, 1, [X, X], [0, 0]).gates
        [Gate('X', 1, [0, 1, 1, 0]), Gate('X', 1, [0, 1, 1, 0])]
        """
        return self._gates

    def __repr__(self):
        """
        >>> from discopy.gates import CX
        >>> Circuit(2, 2, [CX, CX], [0, 0])  # doctest: +ELLIPSIS
        Circuit(2, 2, [Gate('CX', 2, [...]), Gate('CX', 2, [...])], [0, 0])
        """
        return "Circuit({}, {}, {}, {})".format(
            len(self.dom), len(self.cod), self.gates, self.offsets)

    def then(self, other):
        """
        >>> from discopy.gates import CX, SWAP
        >>> print(SWAP >> CX)
        SWAP >> CX
        """
        r = super().then(other)
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    def tensor(self, other):
        """
        >>> from discopy.gates import CX, H
        >>> print(CX @ H)
        CX @ Id(1) >> Id(2) @ H
        """
        r = super().tensor(other)
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    def dagger(self):
        """
        >>> from discopy.gates import CX, SWAP
        >>> print((CX >> SWAP).dagger())
        SWAP.dagger() >> CX.dagger()
        """
        r = super().dagger()
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    @staticmethod
    def id(n):
        """
        >>> Circuit.id(2)
        Id(2)
        """
        return Id(n)

    def eval(self):
        """ Evaluates the circuit as a discopy Matrix.

        >>> from discopy.gates import *
        >>> assert np.all((Ket(0, 0) >> SWAP).eval() == Ket(0, 0).eval())
        >>> assert np.all((Ket(0, 1) >> SWAP).eval() == Ket(1, 0).eval())
        >>> assert np.all((Ket(1, 0) >> SWAP).eval() == Ket(0, 1).eval())
        >>> assert np.all((Ket(1, 1) >> SWAP).eval() == Ket(1, 1).eval())
        >>> assert np.all((Ket(0, 0) >> CX).eval() == Ket(0, 0).eval())
        >>> assert np.all((Ket(0, 1) >> CX).eval() == Ket(0, 1).eval())
        >>> assert np.all((Ket(1, 0) >> CX).eval() == Ket(1, 1).eval())
        >>> assert np.all((Ket(1, 1) >> CX).eval() == Ket(1, 0).eval())
        >>> for U in [SWAP, X, Y, Z, S >> S, CX >> CX >> CX]:
        ...     assert np.all(
        ...         (U >> U.dagger()).eval() == Id(len(U.dom)).eval())
        >>> for U in [H, T >> T >> T >> T]:
        ...     m, id_n = (U >> U.dagger()).eval(), Id(len(U.dom)).eval()
        ...     assert np.allclose(m.array, id_n.array)
        """
        F_eval = MatrixFunctor({Ty(1): 2}, Quiver(lambda g: g.array))
        return F_eval(self)

    def measure(self):
        """
        Applies the Born rule and outputs a stochastic matrix.
        The input maybe any circuit c, the output will be a numpy array
        with shape len(c.dom @ c.cod) * (2, )

        >>> from discopy.gates import Ket, Bra, CX, H, Rx, sqrt
        >>> c = Circuit(2, 2, [sqrt(2), H, Rx(0.5), CX], [0, 0, 1, 0])
        >>> m = c.measure()
        >>> list(np.round(m[0, 0].flatten()))
        [0.0, 1.0, 1.0, 0.0]
        >>> assert np.all((Ket(0, 0) >> c).measure() == m[0, 0])
        >>> assert np.all((c >> Bra(0, 1)).measure() == m[:, :, 0, 1])
        >>> assert (Ket(1, 0) >> c >> Bra(0, 1)).measure() == m[1, 0, 0, 1]
        """
        from discopy.gates import Bra, Ket

        def BornRule(matrix):
            return np.absolute(matrix.array ** 2)

        def bitstring(i, n):
            return map(int, '{{:0{}b}}'.format(n).format(i))
        process = self.eval()
        states, effects = [], []
        states = [Ket(*bitstring(i, len(self.dom))).eval()
                  for i in range(2 ** len(self.dom))]
        effects = [Bra(*bitstring(j, len(self.cod))).eval()
                   for j in range(2 ** len(self.cod))]
        array = np.zeros(len(self.dom + self.cod) * (2, ))
        for state in states if self.dom else [Matrix.id(1)]:
            for effect in effects if self.cod else [Matrix.id(1)]:
                scalar = BornRule(state >> process >> effect)
                array += scalar * (state.dagger() >> effect.dagger()).array
        return array

    def to_tk(self):
        """ Returns a pytket circuit.

        >>> from discopy.gates import SWAP, CX, Rx
        >>> c = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1]).to_tk()
        >>> list(c)
        [SWAP q[0], q[1];, Rx(0.25PI) q[1];, CX q[1], q[2];]
        """
        import pytket as tk
        from discopy.gates import Rx, Rz
        c = tk.Circuit(len(self.dom))
        for g, n in zip(self.gates, self.offsets):
            if isinstance(g, Rx):
                c.Rx(g.data['phase'], *(n + i for i in range(len(g.dom))))
            elif isinstance(g, Rz):
                c.Rz(g.data['phase'], *(n + i for i in range(len(g.dom))))
            else:
                c.__getattribute__(g.name)(*(n + i for i in range(len(g.dom))))
        return c


def from_tk(c):
    """ Takes a pytket circuit and returns a planar circuit,
    SWAP gates are introduced when applying gates to non-adjacent qubits.

    >>> from discopy.gates import SWAP, CX, Rx
    >>> c1 = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1])
    >>> c2 = from_tk(c1.to_tk())
    >>> assert c1 == c2  # This works as long as there are no interchangers!
    """
    from discopy.gates import SWAP, CX, H, S, T, X, Y, Z, Rx, Rz

    def gates_from_tk(g):
        name = g.op.get_type().name
        if name == 'Rx':
            return Rx(g.op.get_params()[0])
        if name == 'Rz':
            return Rz(g.op.get_params()[0])
        return {g.name: g for g in [SWAP, CX, H, S, T, X, Y, Z]}[name]
    gates, offsets = [], []
    for g in c.get_commands():
        i0 = g.qubits[0].index[0]
        for i, q in enumerate(g.qubits[1:]):
            if q.index[0] == i0 + i + 1:
                break  # gate applies to adjacent qubit already
            elif q.index[0] < i0 + i + 1:
                for j in range(q.index, i0 + i):
                    gates.append(SWAP)
                    offsets.append(j)
                if q.index <= i0:
                    i0 -= 1  # we just swapped q to the right of q0
            elif q.index > i0 + i + 1:
                for j in range(q.index - i0 + i - 1):
                    gates.append(SWAP)
                    offsets.append(q.index - j - 1)
        gates.append(gates_from_tk(g))
        offsets.append(i0)
    return Circuit(c.n_qubits, c.n_qubits, gates, offsets)


class Id(Circuit):
    """ Implements identity circuit on n qubits.

    >>> from discopy.gates import SWAP, CX, H, T
    >>> c = CX @ H >> T @ SWAP
    >>> assert Id(3) >> c == c == c >> Id(3)
    """
    def __init__(self, n):
        """
        >>> assert Circuit.id(42) == Id(42) == Circuit(42, 42, [], [])
        """
        if isinstance(n, PRO):
            n = len(n)
        Diagram.__init__(self, PRO(n), PRO(n), [], [])

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(len(self.dom))

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)


def Euler(a, b, c):
    """ Returns a 1-qubit Euler decomposition with angles 2 * pi * a, b, c.

    >>> from discopy.gates import X, Y, Z
    >>> print(Euler(0.1, 0.2, 0.3))
    Rx(0.1) >> Rz(0.2) >> Rx(0.3)
    >>> assert np.all(np.round(Euler(0.5, 0, 0).eval().array) == X.array)
    >>> assert np.all(np.round(Euler(0, 0.5, 0).eval().array) == Z.array)
    >>> assert np.all(np.round(Euler(0, 0, 0.5).eval().array) == X.array)
    >>> assert np.all(
    ...     1j * np.round(Euler(0.5, 0.5, 0).eval().array) == Y.array)
    """
    from discopy.gates import Rx, Rz
    return Rx(a) >> Rz(b) >> Rx(c)


def random(n_qubits, depth=3, gateset=[], seed=None):
    """ Returns a random Euler decomposition if n_qubits == 1,
    otherwise returns a random tiling with the given depth and gateset.

    >>> from discopy.gates import H, T, CX, Rz, Rx
    >>> c = random(1, seed=420)
    >>> print(c)  # doctest: +ELLIPSIS
    Rx(0.026...) >> Rz(0.781...) >> Rx(0.272...)
    >>> array = (c >> c.dagger()).eval().array
    >>> assert np.all(np.round(array) == np.identity(2))
    >>> print(random(2, 2, gateset=[CX, H, T], seed=420))
    CX >> T @ Id(1) >> Id(1) @ T
    >>> print(random(3, 2, gateset=[CX, H, T], seed=420))
    CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
    >>> print(random(2, 1, gateset=[Rz, Rx], seed=420))
    Rz(0.6731171219152886) @ Id(1) >> Id(1) @ Rx(0.2726063832840899)
    """
    from discopy.gates import Rx, Rz
    import random
    if seed:
        random.seed(seed)
    if n_qubits == 1:
        assert depth == 3
        return Euler(random.random(), random.random(), random.random())
    gateset_1 = [g for g in gateset if g is Rx or g is Rz or len(g.dom) == 1]
    U = Id(n_qubits)
    for d in range(depth):
        L, affected = Id(0), 0
        while affected <= n_qubits - 2:
            gate = random.choice(gateset)
            if gate is Rx or gate is Rz:
                gate = gate(random.random())
            L = L @ gate
            affected += len(gate.dom)
        if n_qubits - affected == 1:
            gate = random.choice(gateset_1)
            if gate is Rx or gate is Rz:
                gate = gate(random.random())
            L = L @ gate
        U = U >> L
    return U


class CircuitFunctor(MonoidalFunctor):
    """ Implements funtors from monoidal categories to circuits

    >>> from discopy.gates import SWAP, Rx, CX
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
    >>> d = (f @ Diagram.id(z)
    ...       >> Diagram.id(y) @ g @ Diagram.id(z)
    ...       >> Diagram.id(y) @ h)
    >>> ob = {x: 2, y: 1, z: 1}
    >>> ar = {f: SWAP, g: Rx(0.25), h: CX}
    >>> F = CircuitFunctor(ob, ar)
    >>> print(F(d))
    SWAP @ Id(1) >> Id(1) @ Rx(0.25) @ Id(1) >> Id(1) @ CX
    """
    def __init__(self, ob, ar):
        """
        >>> F = CircuitFunctor({}, {})
        """
        super().__init__({x: PRO(y) for x, y in ob.items()}, ar)

    def __repr__(self):
        """
        >>> CircuitFunctor({}, {})
        CircuitFunctor(ob={}, ar={})
        """
        return "CircuitFunctor(ob={}, ar={})".format(
            repr({x: len(y) for x, y in self.ob.items()}), repr(self.ar))

    def __call__(self, d):
        """
        >>> x = Ty('x')
        >>> F = CircuitFunctor({x: 1}, {})
        >>> assert isinstance(F(Diagram.id(x)), Circuit)
        """
        r = super().__call__(d)
        if isinstance(d, Diagram):
            return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)
        return r
