# -*- coding: utf-8 -*-

"""
Implements quantum circuits as diagrams and circuit-valued monoidal functors.

>>> n = Ty('n')
>>> Alice = Box('Alice', Ty(), n)
>>> loves = Box('loves', n, n)
>>> Bob = Box('Bob', n, Ty())
>>> ob, ar = {n: 1}, {Alice: Ket(0), loves: X, Bob: Bra(1)}
>>> F = CircuitFunctor(ob, ar)
>>> print(F(Alice >> loves >> Bob))
Ket(0) >> X >> Bra(1)
>>> assert F(Alice >> loves >> Bob).eval()
"""

import random as rand

from discopy import messages
from discopy.cat import Quiver
from discopy.monoidal import InterchangerError
from discopy.rigid import Ob, Ty, PRO, Box, Diagram, Functor
from discopy.tensor import np, Dim, Tensor, TensorFunctor


class Circuit(Diagram):
    """
    Implements quantum circuits as diagrams.
    """
    @staticmethod
    def _upgrade(diagram):
        """
        Takes a diagram and returns a circuit.
        """
        return Circuit(len(diagram.dom), len(diagram.cod),
                       diagram.boxes, diagram.offsets, diagram.layers)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        """
        >>> c = Circuit(2, 2, [CX, CX], [0, 0])
        """
        super().__init__(PRO(dom), PRO(cod), boxes, offsets, layers)

    def __repr__(self):
        """
        >>> Circuit(2, 2, [CX, CX], [0, 0])  # doctest: +ELLIPSIS
        Circuit(dom=PRO(2), cod=PRO(2), ...)
        >>> Circuit(2, 2, [CX, CX], [0, 0])  # doctest: +ELLIPSIS
        Circuit(..., boxes=[Gate('CX', ...), Gate('CX', ...)], offsets=[0, 0])
        >>> Circuit(2, 2, [CX], [0])  # doctest: +ELLIPSIS
        Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        >>> Circuit(2, 2, [], [])
        Id(2)
        """
        return super().__repr__().replace('Diagram', 'Circuit')

    def draw(self, draw_types=False, **params):
        return super().draw(**dict(params, draw_types=draw_types))

    @staticmethod
    def id(x):
        """
        >>> Circuit.id(2)
        Id(2)
        """
        return Id(x)

    @staticmethod
    def cups(left, right):
        """
        >>> bell_state = Circuit.cups(PRO(1), PRO(1))
        >>> print(bell_state)
        CX >> H @ Id(1) >> Id(1) @ sqrt(2) @ Id(1) >> Bra(0, 0)
        >>> assert np.allclose(bell_state.eval().array, [[1, 0], [0, 1]])

        >>> double_bell_state = Circuit.cups(PRO(2), PRO(2))
        >>> print('\\n>> '.join(str(layer) for layer in double_bell_state))
        Id(1) @ CX @ Id(1)
        >> Id(1) @ H @ Id(2)
        >> Id(2) @ sqrt(2) @ Id(2)
        >> Id(1) @ Bra(0, 0) @ Id(1)
        >> CX
        >> H @ Id(1)
        >> Id(1) @ sqrt(2) @ Id(1)
        >> Bra(0, 0)
        """
        if not isinstance(left, PRO):
            raise TypeError(messages.type_err(PRO, left))
        if not isinstance(right, PRO):
            raise TypeError(messages.type_err(PRO, right))
        result = Id(left @ right)
        cup = CX >> H @ sqrt(2) @ Id(1) >> Bra(0, 0)
        for i in range(1, len(left) + 1):
            result = result >> Id(len(left) - i) @ cup @ Id(len(left) - i)
        return result

    @staticmethod
    def caps(left, right):
        """
        >>> bell_effect = Circuit.caps(PRO(1), PRO(1))
        >>> print(bell_effect)
        Ket(0, 0) >> Id(1) @ sqrt(2) @ Id(1) >> H @ Id(1) >> CX
        >>> assert np.allclose(bell_effect.eval().array, [[1, 0], [0, 1]])

        >>> double_bell_effect = Circuit.caps(PRO(2), PRO(2))
        >>> print('\\n>> '.join(str(layer) for layer in double_bell_effect))
        Ket(0, 0)
        >> Id(1) @ sqrt(2) @ Id(1)
        >> H @ Id(1)
        >> CX
        >> Id(1) @ Ket(0, 0) @ Id(1)
        >> Id(2) @ sqrt(2) @ Id(2)
        >> Id(1) @ H @ Id(2)
        >> Id(1) @ CX @ Id(1)
        """
        return Circuit.cups(left, right).dagger()

    def eval(self):
        """
        Evaluates the circuit as a discopy Tensor.

        Returns
        -------
        tensor : :class:`discopy.tensor.Tensor`
            with complex amplitudes as entries.

        Examples
        --------
        >>> state, isometry = Ket(1, 1), Id(1) @ Bra(0)
        >>> assert state.eval() >> isometry.eval()\\
        ...     == (state >> isometry).eval()
        """
        return TensorFunctor({Ty(1): 2}, Quiver(lambda g: g.array))(self)

    def interchange(self, i, j, left=False):
        """
        Implements naturality of single kets with respect to the symmetry.

        >>> circuit = sqrt(2) @ Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(', '.join(map(str, circuit.interchange(0, 3).boxes)))
        Ket(1, 0), CX, Ket(0), sqrt(2)
        >>> print(', '.join(map(str, circuit.interchange(3, 0).boxes)))
        Ket(0), sqrt(2), Ket(1, 0), CX, SWAP
        >>> print(', '.join(map(str,
        ...                     circuit.interchange(3, 0, left=True).boxes)))
        Ket(0), sqrt(2), Ket(1, 0), CX, SWAP
        """
        if i == j + 1 and isinstance(self.boxes[i], Ket)\
                and len(self.boxes[i].bitstring) == 1:
            try:
                return super().interchange(i, j, left=left)
            except InterchangerError:
                left_wires, ket, right_wires = self.layers[i]
                if left:
                    layer = Id(len(left_wires) + 1) @ ket\
                        @ Id(len(right_wires) - 1)\
                        >> Id(len(left_wires)) @ SWAP\
                        @ Id(len(right_wires) - 1)
                    return (self[:i] >> layer >> self[i + 1:])\
                        .interchange(i, j, left=left)
                layer = Id(len(left_wires) - 1) @ ket\
                    @ Id(len(right_wires) + 1)\
                    >> Id(len(left_wires) - 1) @ SWAP\
                    @ Id(len(right_wires))
                return (self[:i] >> layer >> self[i + 1:])\
                    .interchange(i, j, left=left)
        else:
            return super().interchange(i, j, left=left)

    def normalize(self, _dagger=False):
        """
        Multiplies all the scalars in the diagram.
        Moves the kets to the top of the diagram, adding swaps if necessary.
        Fuses them into preparation layers.
        Moves the bras to the bottom of the diagram,
        Fuses them into meaurement layers.

        >>> circuit = sqrt(2) @ Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> for step in circuit.normalize():
        ...     print(', '.join(map(str, step.boxes)))
        Ket(1, 0), CX, Ket(0), 1.414
        Ket(1), Ket(0), CX, Ket(0), 1.414
        Ket(1), Ket(0), CX, Ket(0), 1.414
        Ket(0), Ket(1), Ket(0), CX, SWAP, 1.414
        Ket(0), Ket(1), Ket(0), 1.414, CX, SWAP
        Ket(0, 1), Ket(0), 1.414, CX, SWAP
        Ket(0, 1, 0), 1.414, CX, SWAP
        """
        def remove_scalars(diagram):
            for i, box in enumerate(diagram.boxes):
                if box.dom == box.cod == PRO():
                    return diagram[:i] >> diagram[i + 1:], box.array[0]
            return diagram, None

        def find_ket(diagram):
            boxes, offsets = diagram.boxes, diagram.offsets
            for i in range(len(diagram) - 1):
                if isinstance(boxes[i], Ket) and isinstance(boxes[i + 1], Ket)\
                        and offsets[i + 1] == offsets[i] + len(boxes[i].cod):
                    return i
            return None

        def fuse_kets(diagram, i):
            boxes, offsets = diagram.boxes, diagram.offsets
            ket = Ket(*(boxes[i].bitstring + boxes[i + 1].bitstring))
            return Circuit(len(diagram.dom), len(diagram.cod),
                           boxes[:i] + [ket] + boxes[i + 2:],
                           offsets[:i + 1] + offsets[i + 2:])

        def unfuse(ket):
            if not isinstance(ket, Ket):
                return ket
            result = Id(0)
            for bit in ket.bitstring:
                result = result @ Ket(bit)
            return result

        diagram = self
        # step 0: multiply scalars to the right of the diagram
        if not _dagger:
            scalar = 1
            while True:
                diagram, number = remove_scalars(diagram)
                if number is None:
                    break
                scalar = scalar * number
                yield diagram @ Gate('{0:.3f}'.format(scalar), 0, [scalar])
            diagram = diagram @ Gate('{0:.3f}'.format(scalar), 0, [scalar])

        # step 1: unfuse all kets
        before = diagram
        diagram = CircuitFunctor(ob=Quiver(len), ar=Quiver(unfuse))(diagram)
        if diagram != before:
            yield diagram

        # step 2: move kets to the bottom of the diagram by foliating
        ket_count = sum([1 if isinstance(box, Ket) else 0
                         for box in diagram.boxes])
        gen = diagram.foliate()
        for _ in range(ket_count):
            diagram = next(gen)
            yield diagram

        # step 4: fuse kets
        while True:
            fusable = find_ket(diagram)
            if fusable is None:
                break
            diagram = fuse_kets(diagram, fusable)
            yield diagram

        # step 5: repeat for bras using dagger
        if not _dagger:
            for _diagram in diagram.dagger().normalize(_dagger=True):
                yield _diagram.dagger()

    def normal_form(self):
        """
        Rewrites self into a circuit of the form:
        kets >> unitary >> bras >> scalar.
        Where 'kets' is a slice of preparation layers,
        'bras' is a slice of measurement layers,
        and 'scalar' is the renormalization factor.

        >>> caps = Circuit.caps(PRO(2), PRO(2))
        >>> cups = Circuit.cups(PRO(2), PRO(2))
        >>> snake = caps @ Id(2) >> Id(2) @ cups
        >>> snake_nf = snake.normal_form()
        >>> assert snake_nf.boxes[0] == Ket(0, 0, 0, 0)
        >>> assert snake_nf.boxes[-2] == Bra(0, 0, 0, 0)
        >>> assert snake_nf.boxes[-1].name == '4.000'
        """
        *_, result = list(self.normalize()) or [self]
        return result

    @staticmethod
    def random(n_qubits, depth=3, gateset=None, seed=None):
        """ Returns a random Euler decomposition if n_qubits == 1,
        otherwise returns a random tiling with the given depth and gateset.

        >>> c = Circuit.random(1, seed=420)
        >>> print(c)  # doctest: +ELLIPSIS
        Rx(0.026... >> Rz(0.781... >> Rx(0.272...
        >>> print(Circuit.random(2, 2, gateset=[CX, H, T], seed=420))
        CX >> T @ Id(1) >> Id(1) @ T
        >>> print(Circuit.random(3, 2, gateset=[CX, H, T], seed=420))
        CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
        >>> print(Circuit.random(2, 1, gateset=[Rz, Rx], seed=420))
        Rz(0.6731171219152886) @ Id(1) >> Id(1) @ Rx(0.2726063832840899)
        """
        if seed is not None:
            rand.seed(seed)
        if n_qubits == 1:
            return Rx(rand.random()) >> Rz(rand.random()) >> Rx(rand.random())
        result = Id(n_qubits)
        for _ in range(depth):
            line, n_affected = Id(0), 0
            while n_affected < n_qubits:
                gate = rand.choice(
                    gateset if n_qubits - n_affected > 1 else [
                        g for g in gateset
                        if g is Rx or g is Rz or len(g.dom) == 1])
                if gate is Rx or gate is Rz:
                    gate = gate(rand.random())
                line = line @ gate
                n_affected += len(gate.dom)
            result = result >> line
        return result

    def measure(self):
        """
        Measures a circuit on the computational basis.

        Returns
        -------
        array : np.ndarray
            with real entries and the same shape as :code:`self.eval().array`.

        Examples
        --------
        >>> m = X.measure()
        >>> list(np.round(m.flatten()))
        [0.0, 1.0, 1.0, 0.0]
        >>> assert (Ket(0) >> X >> Bra(1)).measure() == m[0, 1]
        """
        def bitstring(i, length):
            return map(int, '{{:0{}b}}'.format(length).format(i))
        process = self.eval()
        states, effects = [], []
        states = [Ket(*bitstring(i, len(self.dom))).eval()
                  for i in range(2 ** len(self.dom))]
        effects = [Bra(*bitstring(j, len(self.cod))).eval()
                   for j in range(2 ** len(self.cod))]
        array = np.zeros(len(self.dom + self.cod) * (2, ))
        for state in states if self.dom else [Tensor.id(1)]:
            for effect in effects if self.cod else [Tensor.id(1)]:
                scalar = np.absolute((state >> process >> effect).array) ** 2
                array += scalar * (state.dagger() >> effect.dagger()).array
        return array

    def to_tk(self):
        """
        Returns
        -------
        tk_circuit : pytket.Circuit
            A :class:`pytket.Circuit`.

        Note
        ----
        * No measurements are performed.
        * SWAP gates are treated as logical swaps.
        * If the circuit contains scalars or a :class:`Bra`,
          then :code:`tk_circuit` will hold attributes
          :code:`post_selection` and :code:`scalar`.

        Examples
        --------
        >>> circuit0 = H @ Rx(0.5) >> CX
        >>> print(list(circuit0.to_tk()))
        [H q[0];, Rx(1*PI) q[1];, CX q[0], q[1];]

        >>> circuit1 = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(list(circuit1.to_tk()))
        [X q[0];, CX q[0], q[2];]
        >>> circuit2 = Circuit.from_tk(circuit1.to_tk())
        >>> print(circuit2)
        X @ Id(2) >> Id(1) @ SWAP >> CX @ Id(1) >> Id(1) @ SWAP
        >>> print(list(circuit2.to_tk()))
        [X q[0];, CX q[0], q[2];]

        >>> circuit = Ket(0, 0)\\
        ...     >> sqrt(2) @ Id(2)\\
        ...     >> H @ Id(1)\\
        ...     >> Id(1) @ X\\
        ...     >> CX\\
        ...     >> Id(1) @ Bra(0)
        >>> tk_circ = circuit.to_tk()
        >>> print(list(tk_circ))
        [H q[0];, X q[1];, CX q[0], q[1];]
        >>> print(tk_circ.post_selection)
        {1: 0}
        >>> print(np.round(abs(tk_circ.scalar) ** 2))
        2.0
        """
        from discopy.tk_interface import to_tk
        return to_tk(self)

    @staticmethod
    def from_tk(tk_circuit):
        """
        Parameters
        ----------
        tk_circuit : pytket.Circuit
            A pytket.Circuit, potentially with :code:`scalar` and
            :code:`post_selection` attributes.

        Returns
        -------
        circuit : :class:`Circuit`
            Such that :code:`Circuit.from_tk(circuit.to_tk()) == circuit`.

        Note
        ----
        * SWAP gates are introduced when applying gates to non-adjacent qubits.

        Examples
        --------
        >>> c1 = Circuit(2, 2, [Rz(0.5), Rx(0.25), CX], [0, 1, 0])
        >>> c2 = Circuit.from_tk(c1.to_tk())
        >>> assert c1.normal_form() == c2.normal_form()

        >>> import pytket as tk
        >>> tk_GHZ = tk.Circuit(3).H(1).CX(1, 2).CX(1, 0)
        >>> print(Circuit.from_tk(tk_GHZ))
        Id(1) @ H @ Id(1)\\
          >> Id(1) @ CX\\
          >> SWAP @ Id(1)\\
          >> CX @ Id(1)\\
          >> SWAP @ Id(1)
        >>> circuit = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(Circuit.from_tk(circuit.to_tk()))
        X @ Id(2) >> Id(1) @ SWAP >> CX @ Id(1) >> Id(1) @ SWAP

        >>> bell_state = Circuit.caps(PRO(1), PRO(1))
        >>> bell_effect = bell_state[::-1]
        >>> circuit = bell_state @ Id(1) >> Id(1) @ bell_effect >> Bra(0)
        >>> print(Circuit.from_tk(circuit.to_tk()))
        H @ Id(2)\\
          >> CX @ Id(1)\\
          >> Id(1) @ CX\\
          >> Id(1) @ H @ Id(1)\\
          >> Id(2) @ Bra(0)\\
          >> Id(1) @ Bra(0)\\
          >> Bra(0)\\
          >> scalar(2.000)
        """
        from discopy.tk_interface import from_tk
        return from_tk(tk_circuit)

    def get_counts(self, backend, **params):
        """
        Parameters
        ----------
        backend : pytket.Backend
         Backend on which to run the circuit.
        n_shots : int, optional
         Number of shots, default is :code:`2**10`.
        measure_all : bool, optional
         Whether to measure all qubits, default is :code:`True`.
        normalize : bool, optional
         Whether to normalize the counts, default is :code:`True`.
        post_select : bool, optional
         Whether to perform post-selection, default is :code:`True`.
        scale : bool, optional
         Whether to scale the output, default is :code:`True`.
        seed : int, optional
         Seed to feed the backend, default is :code:`None`.

        Returns
        -------
        tensor : :class:`discopy.tensor.Tensor`
         Of dimension :code:`n_qubits * (2, )` for :code:`n_qubits` the
         number of post-selected qubits.

        Examples
        --------
        >>> from pytket.backends.ibm import AerBackend
        >>> backend = AerBackend()
        >>> circuit = H @ Id(1) >> CX >> Id(1) @ Bra(0)
        >>> circuit.get_counts(backend, seed=42)  # doctest: +ELLIPSIS
        Tensor(dom=Dim(1), cod=Dim(2), array=[0.49..., 0.0])
        >>> scaled_bell = Circuit.caps(PRO(1), PRO(1))
        >>> snake = scaled_bell @ Id(1) >> Id(1) @ scaled_bell[::-1]
        >>> assert np.all(
        ...     np.round(snake.get_counts(backend, seed=42).array)
        ...     == np.round((Ket(0) >> snake).measure()))
        """
        from discopy.tk_interface import get_counts
        return get_counts(self, backend, **params)


class Id(Circuit):
    """ Implements identity circuit on n qubits.

    >>> c = CX @ H >> T @ SWAP
    >>> assert Id(3) >> c == c == c >> Id(3)
    """
    def __init__(self, n_qubits):
        """
        >>> assert Circuit.id(42) == Id(42) == Circuit(42, 42, [], [])
        """
        if isinstance(n_qubits, PRO):
            n_qubits = len(n_qubits)
        super().__init__(n_qubits, n_qubits, [], [])

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


class Gate(Box, Circuit):
    """ Implements quantum gates as boxes in a circuit diagram.

    >>> CX
    Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    """
    def __init__(self, name, n_qubits, array=None, data=None, _dagger=False):
        """
        >>> g = CX
        >>> assert g.dom == g.cod == PRO(2)
        """
        if array is not None:
            self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits),
                     data=data, _dagger=_dagger)
        Circuit.__init__(self, n_qubits, n_qubits, [self], [0])

    @property
    def array(self):
        """
        >>> list(X.array.flatten())
        [0, 1, 1, 0]
        """
        return self._array

    def __repr__(self):
        """
        >>> CX
        Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        >>> X.dagger()
        Gate('X', 1, [0, 1, 1, 0])
        >>> Y.dagger()
        Gate('Y', 1, [0j, (-0-1j), 1j, 0j]).dagger()
        """
        if self._dagger:
            return repr(self.dagger()) + '.dagger()'
        return "Gate({}, {}, {}{})".format(
            repr(self.name), len(self.dom), list(self.array.flatten()),
            ', data=' + repr(self.data) if self.data else '')

    def dagger(self):
        """
        >>> print(CX.dagger())
        CX
        >>> print(Y.dagger())
        Y[::-1]
        >>> assert Y.eval().dagger() == Y.dagger().eval()
        """
        return Gate(
            self.name, len(self.dom), self.array, data=self.data,
            _dagger=None if self._dagger is None else not self._dagger)


class Ket(Box, Circuit):
    """ Implements ket for a given bitstring.

    >>> Ket(1, 1, 0).eval()
    Tensor(dom=Dim(1), cod=Dim(2, 2, 2), array=[0, 0, 0, 0, 0, 0, 1, 0])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Ket(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Ket({})'.format(', '.join(map(str, bitstring))),
                     PRO(0), PRO(len(bitstring)))
        Circuit.__init__(self, 0, len(bitstring), [self], [0])

    def tensor(self, other):
        """
        When two Kets are tensored together, they yield one big Ket with the
        concatenation of their bitstrings.

        >>> Ket(0, 1, 0) @ Ket(1, 0)
        Ket(0, 1, 0, 1, 0)
        >>> assert isinstance(Ket(1) @ Id(1) @ Ket(1, 0), Circuit)
        """
        if isinstance(other, Ket):
            return Ket(*(self.bitstring + other.bitstring))
        return super().tensor(other)

    def __repr__(self):
        """
        >>> Ket(1, 1, 0)
        Ket(1, 1, 0)
        """
        return self.name

    def dagger(self):
        """
        >>> Ket(0, 1).dagger()
        Bra(0, 1)
        """
        return Bra(*self.bitstring)

    @property
    def array(self):
        """
        >>> Ket(0).eval()
        Tensor(dom=Dim(1), cod=Dim(2), array=[1, 0])
        >>> Ket(0, 1).eval()
        Tensor(dom=Dim(1), cod=Dim(2, 2), array=[0, 1, 0, 0])
        """
        tensor = Tensor(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            tensor = tensor @ Tensor(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return tensor.array


class Bra(Box, Circuit):
    """ Implements bra for a given bitstring.

    >>> Bra(1, 1, 0).eval()
    Tensor(dom=Dim(2, 2, 2), cod=Dim(1), array=[0, 0, 0, 0, 0, 0, 1, 0])
    >>> assert all((Bra(x, y, z) << Ket(x, y, z)).eval() == 1
    ...            for x in [0, 1] for y in [0, 1] for z in [0, 1])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Bra(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Bra({})'.format(', '.join(map(str, bitstring))),
                     PRO(len(bitstring)), PRO(0))
        Circuit.__init__(self, len(bitstring), 0, [self], [0])

    def __repr__(self):
        """
        >>> Bra(1, 1, 0)
        Bra(1, 1, 0)
        """
        return self.name

    def tensor(self, other):
        """
        When two Bras are tensored together, they yield one big Bra with the
        concatenation of their bitstrings.

        >>> Bra(0, 1, 0) @ Bra(1, 0)
        Bra(0, 1, 0, 1, 0)
        >>> print(Bra(0) @ X)
        Bra(0) @ Id(1) >> X
        """
        if isinstance(other, Bra):
            return Bra(*(self.bitstring + other.bitstring))
        return super().tensor(other)

    def dagger(self):
        """
        >>> Bra(0, 1).dagger()
        Ket(0, 1)
        """
        return Ket(*self.bitstring)

    @property
    def array(self):
        """
        >>> Bra(0).eval()
        Tensor(dom=Dim(2), cod=Dim(1), array=[1, 0])
        >>> Bra(0, 1).eval()
        Tensor(dom=Dim(2, 2), cod=Dim(1), array=[0, 1, 0, 0])
        """
        return Ket(*self.bitstring).array


class Rz(Gate):
    """
    >>> assert np.all(Rz(0).array == np.identity(2))
    >>> assert np.allclose(Rz(0.5).array, Z.array)
    >>> assert np.allclose(Rz(0.25).array, S.array)
    >>> assert np.allclose(Rz(0.125).array, T.array)
    """
    def __init__(self, phase):
        """
        >>> Rz(0.25)
        Rz(0.25)
        """
        self._phase = phase
        super().__init__('Rz', 1)

    @property
    def phase(self):
        """
        >>> Rz(0.25).phase
        0.25
        """
        return self._phase

    @property
    def name(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125)) == Rz(0.125).name
        """
        return 'Rz({})'.format(self.phase)

    def __repr__(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rz(0.5).dagger().eval() == Rz(0.5).eval().dagger()
        """
        return Rz(-self.phase)

    @property
    def array(self):
        """
        >>> assert np.allclose(Rz(0.5).array, Z.array)
        """
        theta = 2 * np.pi * self.phase
        return np.array([[1, 0], [0, np.exp(1j * theta)]])


class Rx(Gate):
    """
    >>> assert np.all(np.round(Rx(0.5).array) == X.array)
    """
    def __init__(self, phase):
        """
        >>> Rx(0.25)
        Rx(0.25)
        """
        self._phase = phase
        super().__init__('Rx', 1)

    @property
    def phase(self):
        """
        >>> Rx(0.25).phase
        0.25
        """
        return self._phase

    @property
    def name(self):
        """
        >>> assert str(Rx(0.125)) == Rx(0.125).name
        """
        return 'Rx({})'.format(self.phase)

    def __repr__(self):
        """
        >>> assert str(Rx(0.125)) == repr(Rx(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rx(0.5).dagger().eval() == Rx(0.5).eval().dagger()
        """
        return Rx(-self.phase)

    @property
    def array(self):
        half_theta = np.pi * self.phase
        global_phase = np.exp(1j * half_theta)
        sin, cos = np.sin(half_theta), np.cos(half_theta)
        return global_phase * np.array([[cos, -1j * sin], [-1j * sin, cos]])


class CircuitFunctor(Functor):
    """ Implements funtors from monoidal categories to circuits

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
        super().__init__(ob, ar, ob_factory=PRO, ar_factory=Circuit)

    def __repr__(self):
        """
        >>> CircuitFunctor({}, {})
        CircuitFunctor(ob={}, ar={})
        """
        return "CircuitFunctor(ob={}, ar={})".format(
            repr(self.ob), repr(self.ar))

    def __call__(self, diagram):
        """
        >>> x = Ty('x')
        >>> F = CircuitFunctor({x: 1}, {})
        >>> assert isinstance(F(Diagram.id(x)), Circuit)
        """
        if isinstance(diagram, Ty):
            return PRO(len(super().__call__(diagram)))
        if isinstance(diagram, Ob) and not diagram.z:
            return PRO(self.ob[Ty(diagram.name)])
        if isinstance(diagram, Diagram):
            return Circuit._upgrade(super().__call__(diagram))
        return super().__call__(diagram)


def sqrt(real):
    """
    >>> sqrt(2)  # doctest: +ELLIPSIS
    Gate('sqrt(2)', 0, [1.41...])
    """
    return Gate('sqrt({})'.format(real), 0, np.sqrt(real), _dagger=None)


def scalar(complex):
    return Gate('scalar({:.3f})'.format(complex), 0, complex,
                _dagger=None if np.conjugate(complex) == complex else False)


SWAP = Gate('SWAP', 2, [1, 0, 0, 0,
                        0, 0, 1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1], _dagger=None)
CX = Gate('CX', 2, [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0], _dagger=None)
H = Gate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]), _dagger=None)
S = Gate('S', 1, [1, 0, 0, 1j])
T = Gate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = Gate('X', 1, [0, 1, 1, 0], _dagger=None)
Y = Gate('Y', 1, [0, -1j, 1j, 0])
Z = Gate('Z', 1, [1, 0, 0, -1], _dagger=None)
