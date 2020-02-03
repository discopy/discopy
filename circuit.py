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

import pytket as tk
from pytket.circuit import UnitID

from discopy import moncat, messages
from discopy.cat import Quiver
from discopy.moncat import InterchangerError
from discopy.rigidcat import Ob, Ty, PRO, Box, Diagram, RigidFunctor
from discopy.matrix import np, Dim, Matrix, MatrixFunctor


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
        Evaluates the circuit as a discopy Matrix.

        Returns
        -------
        matrix : `discopy.matrix.Matrix`
            with complex amplitudes as entries.

        Examples
        --------
        >>> state, isometry = Ket(1, 1), Id(1) @ Bra(0)
        >>> assert state.eval() >> isometry.eval()\\
        ...     == (state >> isometry).eval()
        """
        return MatrixFunctor({Ty(1): 2}, Quiver(lambda g: g.array))(self)

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
                else:
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
        for i in range(ket_count):
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
        for state in states if self.dom else [Matrix.id(1)]:
            for effect in effects if self.cod else [Matrix.id(1)]:
                scalar = np.absolute((state >> process >> effect).array) ** 2
                array += scalar * (state.dagger() >> effect.dagger()).array
        return array

    def to_tk(self):
        """
        Returns a pytket circuit. SWAP gates are treated as logical swaps.

        >>> circuit0 = H @ Rx(0.5) >> CX
        >>> print(list(circuit0.to_tk()))
        [H q[0];, Rx(0.5*PI) q[1];, CX q[0], q[1];]

        >>> circuit1 = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(list(circuit1.to_tk()))
        [X q[0];, CX q[0], q[2];]
        >>> circuit2 = Circuit.from_tk(circuit1.to_tk())
        >>> print(circuit2)
        X @ Id(2) >> Id(1) @ SWAP >> CX @ Id(1) >> Id(1) @ SWAP
        >>> print(list(circuit2.to_tk()))
        [X q[0];, CX q[0], q[2];]
        """
        def remove_ket1(box):
            if not isinstance(box, Ket):
                return box
            x_gates = Circuit.id(0)
            for bit in box.bitstring:
                x_gates = x_gates @ (X if bit else Circuit.id(1))
            return Ket(*(len(box.bitstring) * (0, ))) >> x_gates

        def swap(tk_circ, i, j):
            old = UnitID('q', i)
            tmp = UnitID('tmp', 0)
            new = UnitID('q', j)
            tk_circ.rename_units({old: tmp})
            tk_circ.rename_units({new: old})
            tk_circ.rename_units({tmp: new})

        def add_qubit(tk_circ, left, box, right):
            if len(right) > 0:
                renaming = dict()
                for i in range(len(left), tk_circ.n_qubits):
                    old = UnitID('q', i)
                    new = UnitID('q', i + len(box.cod))
                    renaming.update({old: new})
                tk_circ.rename_units(renaming)
            tk_circ.add_blank_wires(len(box.cod))

        def add_gate(tk_circ, box, off):
            qubits = [off + j for j in range(len(box.dom))]
            if isinstance(box, (Rx, Rz)):
                tk_circ.__getattribute__(box.name[:2])(box.phase, *qubits)
            else:
                tk_circ.__getattribute__(box.name)(*qubits)

        def measure_qubit(tk_circ, left, box, right):
            for i, _ in enumerate(box.dom):
                tk_circ.add_bit(UnitID('c', len(tk_circ.bits)))
                tk_circ.Measure(len(left) + i, len(tk_circ.bits) - 1)
            if len(right) > 0:
                renaming = dict()
                for i, _ in enumerate(box.dom):
                    old = UnitID('q', len(left) + i)
                    tmp = UnitID('tmp', i)
                    renaming.update({old: tmp})
                for i, _ in enumerate(right):
                    old = UnitID('q', len(left @ box.dom) + i)
                    new = UnitID('q', len(left) + i)
                    renaming.update({old: new})
                tk_circ.rename_units(renaming)
                renaming = dict()
                for j, _ in enumerate(box.dom):
                    tmp = UnitID('tmp', j)
                    new = UnitID(
                        'q', tk_circ.n_qubits - len(tk_circ.bits) + j)
                    renaming.update({tmp: new})
                tk_circ.rename_units(renaming)
        circuit = CircuitFunctor(ob=Quiver(len), ar=Quiver(remove_ket1))(self)
        if circuit.dom != PRO(0):
            circuit = Ket(*(len(circuit.dom) * (0, ))) >> circuit
        tk_circ = tk.Circuit()
        for left, box, right in circuit.layers:
            if isinstance(box, Ket):
                add_qubit(tk_circ, left, box, right)
            elif isinstance(box, Bra):
                measure_qubit(tk_circ, left, box, right)
            elif box == SWAP:
                swap(tk_circ, len(left), len(left) + 1)
            else:
                add_gate(tk_circ, box, len(left))
        return tk_circ

    @staticmethod
    def from_tk(tk_circuit):
        """ Takes a pytket circuit and returns a planar circuit,
        SWAP gates are introduced when applying gates to non-adjacent qubits.

        >>> c1 = Circuit(2, 2, [Rz(0.5), Rx(0.25), CX], [0, 1, 0])
        >>> c2 = Circuit.from_tk(c1.to_tk())
        >>> assert c1.normal_form() == c2.normal_form()

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
        """
        def box_from_tk(tk_gate):
            name = tk_gate.op.get_type().name
            if name == 'Measure':
                return Bra(0)
            if name == 'Rx':
                return Rx(tk_gate.op.get_params()[0])
            if name == 'Rz':
                return Rz(tk_gate.op.get_params()[0])
            for gate in [SWAP, CX, H, S, T, X, Y, Z]:
                if name == gate.name:
                    return gate
            raise NotImplementedError

        def permute(tk_circuit, tk_gate):
            n_qubits, i_0 = tk_circuit.n_qubits, tk_gate.qubits[0].index[0]
            result = Id(n_qubits)
            for i, qubit in enumerate(tk_gate.qubits[1:]):
                if qubit.index[0] == i_0 + i + 1:
                    continue  # gate applies to adjacent qubit already
                if qubit.index[0] < i_0 + i + 1:
                    for j in range(qubit.index[0], i_0 + i):
                        result = result >> Id(j) @ SWAP @ Id(n_qubits - j - 2)
                    if qubit.index[0] <= i_0:
                        i_0 -= 1
                else:
                    for j in range(qubit.index[0] - i_0 + i - 1):
                        left = qubit.index[0] - j - 1
                        right = n_qubits - left - 2
                        result = result >> Id(left) @ SWAP @ Id(right)
            return i_0, result
        n_qubits, n_bits = tk_circuit.n_qubits, 0
        circuit = Id(n_qubits)
        for tk_gate in tk_circuit.get_commands():
            i_0, permutation = permute(tk_circuit, tk_gate)
            left, right = i_0, n_qubits - i_0 - len(tk_gate.qubits) - n_bits
            box = box_from_tk(tk_gate)
            if isinstance(box, Bra):
                n_bits += 1
            layer = Id(left) @ box @ Id(right)
            if len(permutation) > 0:
                layer = permutation >> layer >> permutation[::-1]
            circuit = circuit >> layer
        return circuit

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
    Matrix(dom=Dim(1), cod=Dim(2, 2, 2), array=[0, 0, 0, 0, 0, 0, 1, 0])
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
        Matrix(dom=Dim(1), cod=Dim(2), array=[1, 0])
        >>> Ket(0, 1).eval()
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[0, 1, 0, 0])
        """
        matrix = Matrix(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            matrix = matrix @ Matrix(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return matrix.array


class Bra(Box, Circuit):
    """ Implements bra for a given bitstring.

    >>> Bra(1, 1, 0).eval()
    Matrix(dom=Dim(2, 2, 2), cod=Dim(1), array=[0, 0, 0, 0, 0, 0, 1, 0])
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
        Matrix(dom=Dim(2), cod=Dim(1), array=[1, 0])
        >>> Bra(0, 1).eval()
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[0, 1, 0, 0])
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


class CircuitFunctor(RigidFunctor):
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
        super().__init__(ob, ar, ob_cls=PRO, ar_cls=Circuit)

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
