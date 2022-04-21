# -*- coding: utf-8 -*-

""" Implements ZX diagrams. """

from discopy import messages, cat, monoidal, rigid, quantum, tensor
from discopy.monoidal import Sum
from discopy.rigid import Functor, PRO
from discopy.quantum.circuit import Circuit, qubit
from discopy.quantum.gates import (
    Bra, Ket, Rz, Rx, Ry, CX, CZ, CRz, CRx, Controlled, format_number)
from discopy.quantum.gates import Scalar as GatesScalar
from math import pi


@monoidal.Diagram.subclass
class Diagram(tensor.Diagram):
    """ ZX Diagram. """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'zx.Diagram')

    @staticmethod
    def swap(left, right):
        left = left if isinstance(left, PRO) else PRO(left)
        right = right if isinstance(right, PRO) else PRO(right)
        return monoidal.Diagram.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)

    @staticmethod
    def permutation(perm, dom=None):
        dom = PRO(len(perm)) if dom is None else dom
        return monoidal.Diagram.permutation(perm, dom, ar_factory=Diagram)

    @staticmethod
    def cups(left, right):
        return rigid.cups(
            left, right, ar_factory=Diagram, cup_factory=lambda *_: Z(2, 0))

    @staticmethod
    def caps(left, right):
        return rigid.caps(
            left, right, ar_factory=Diagram, cap_factory=lambda *_: Z(0, 2))

    def draw(self, **params):
        """ ZX diagrams don't have labels on wires. """
        return super().draw(**dict(params, draw_type_labels=False))

    def grad(self, var, **params):
        """
        Gradient with respect to `var`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

        Returns
        -------
        diagrams : discopy.monoidal.Sum

        Examples
        --------
        >>> from sympy.abc import phi
        >>> assert Z(1, 1, phi).grad(phi) == scalar(pi) @ Z(1, 1, phi + .5)
        """
        return super().grad(var, **params)

    def to_pyzx(self):
        """
        Returns a :class:`pyzx.Graph`.

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert len(graph.vertices()) == 8
        >>> assert (graph.inputs, graph.outputs) == ([0, 1], [6, 7])
        >>> from pyzx import VertexType
        >>> assert graph.type(2) == graph.type(3) == VertexType.Z
        >>> assert graph.phase(2) == 2 * .25 and graph.phase(3) == 2 * .75
        >>> assert graph.type(4) == graph.type(5) == VertexType.X
        >>> assert graph.phase(4) == graph.phase(5) == 2 * .5
        >>> assert graph.graph == {
        ...     0: {2: 1},
        ...     1: {3: 1},
        ...     2: {0: 1, 4: 1, 5: 1},
        ...     3: {1: 1, 4: 1, 5: 1},
        ...     4: {2: 1, 3: 1, 6: 1},
        ...     5: {2: 1, 3: 1, 7: 1},
        ...     6: {4: 1},
        ...     7: {5: 1}}
        """
        from pyzx import Graph, VertexType, EdgeType
        graph, scan = Graph(), []
        for i, _ in enumerate(self.dom):
            node, hadamard = graph.add_vertex(VertexType.BOUNDARY), False
            scan.append((node, hadamard))
            graph.inputs.append(node)
            graph.set_position(node, i, 0)
        for row, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box, Spider):
                node = graph.add_vertex(
                    VertexType.Z if isinstance(box, Z) else VertexType.X,
                    phase=box.phase * 2 if box.phase else None)
                graph.set_position(node, offset, row + 1)
                for i, _ in enumerate(box.dom):
                    source, hadamard = scan[offset + i]
                    etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
                    graph.add_edge((source, node), etype)
                scan = scan[:offset] + len(box.cod) * [(node, False)]\
                    + scan[offset + len(box.dom):]
            elif isinstance(box, Swap):
                scan = scan[:offset] + [scan[offset + 1], scan[offset]]\
                    + scan[offset + 2:]
            elif isinstance(box, Scalar):
                graph.scalar.add_float(box.data)
            elif box == H:
                node, hadamard = scan[offset]
                scan[offset] = (node, not hadamard)
            else:
                raise TypeError(messages.type_err(Box, box))
        for i, _ in enumerate(self.cod):
            target = graph.add_vertex(VertexType.BOUNDARY)
            source, hadamard = scan[i]
            etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
            graph.add_edge((source, target), etype)
            graph.set_position(target, i, len(self) + 1)
            graph.outputs.append(target)
        return graph

    @staticmethod
    def from_pyzx(graph):
        """
        Takes a :class:`pyzx.Graph` returns a :class:`zx.Diagram`.

        Examples
        --------

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert Diagram.from_pyzx(graph) == bialgebra

        Note
        ----

        Raises :code:`ValueError` if either:
        * a boundary node is not in :code:`graph.inputs + graph.outputs`,
        * :code:`set(graph.inputs).intersection(graph.outputs)` is non-empty.
        """
        from pyzx import VertexType, EdgeType

        def node2box(node, n_legs_in, n_legs_out):
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return (Z if graph.type(node) == VertexType.Z else X)(
                n_legs_in, n_legs_out, graph.phase(node) * .5)

        def move(scan, source, target):
            if target < source:
                swaps = Id(target)\
                    @ Diagram.swap(source - target, 1)\
                    @ Id(len(scan) - source - 1)
                scan = scan[:target] + [node]\
                    + scan[target:source] + scan[source + 1:]
            elif target > source:
                swaps = Id(source)\
                    @ Diagram.swap(1, target - source)\
                    @ Id(len(scan) - target - 1)
                scan = scan[:source] + scan[source + 1:target]\
                    + [node] + scan[target:]
            else:
                swaps = Id(len(scan))
            return scan, swaps

        def make_wires_adjacent(scan, diagram, inputs):
            if not inputs:
                return scan, diagram, len(scan)
            offset = scan.index(inputs[0])
            for i, _ in enumerate(inputs[1:]):
                source, target = scan.index(inputs[i + 1]), offset + i + 1
                scan, swaps = move(scan, source, target)
                diagram = diagram >> swaps
            return scan, diagram, offset

        missing_boundary = any(
            graph.type(node) == VertexType.BOUNDARY
            and node not in graph.inputs + graph.outputs
            for node in graph.vertices())
        if missing_boundary:
            raise ValueError
        duplicate_boundary = set(graph.inputs).intersection(graph.outputs)
        if duplicate_boundary:
            raise ValueError
        diagram, scan = Id(len(graph.inputs)), graph.inputs
        for node in [v for v in graph.vertices()
                     if v not in graph.inputs + graph.outputs]:
            inputs = [v for v in graph.neighbors(node) if v < node
                      and v not in graph.outputs or v in graph.inputs]
            inputs.sort(key=scan.index)
            outputs = [v for v in graph.neighbors(node) if v > node
                       and v not in graph.inputs or v in graph.outputs]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            hadamards = Id(0).tensor(*[
                H if graph.edge_type((i, node)) == EdgeType.HADAMARD else Id(1)
                for i in scan[offset: offset + len(inputs)]])
            box = node2box(node, len(inputs), len(outputs))
            diagram = diagram >> Id(offset) @ (hadamards >> box)\
                @ Id(len(diagram.cod) - offset - len(inputs))
            scan = scan[:offset] + len(outputs) * [node]\
                + scan[offset + len(inputs):]
        for target, output in enumerate(graph.outputs):
            node, = graph.neighbors(output)
            etype = graph.edge_type((node, output))
            hadamard = H if etype == EdgeType.HADAMARD else Id(1)
            scan, swaps = move(scan, scan.index(node), target)
            diagram = diagram >> swaps\
                >> Id(target) @ hadamard @ Id(len(scan) - target - 1)
        return diagram


class Id(rigid.Id, Diagram):
    """ Identity ZX diagram. """
    def __init__(self, dom=0):
        super().__init__(PRO(dom))

    def __repr__(self):
        return "Id({})".format(len(self.dom))

    __str__ = __repr__


Diagram.id = Id


class Box(rigid.Box, Diagram):
    """ Box in a ZX diagram. """
    def __init__(self, name, dom, cod, **params):
        if not isinstance(dom, PRO):
            raise TypeError(messages.type_err(PRO, dom))
        if not isinstance(cod, PRO):
            raise TypeError(messages.type_err(PRO, cod))
        rigid.Box.__init__(self, name, dom, cod, **params)
        Diagram.__init__(self, dom, cod, [self], [0])


class Swap(rigid.Swap, Box):
    """ Swap in a ZX diagram. """
    def __init__(self, left, right):
        rigid.Swap.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod)

    def __repr__(self):
        return "SWAP"

    __str__ = __repr__


SWAP = Swap(PRO(1), PRO(1))


class Spider(Box):
    """ Abstract spider box. """
    def __init__(self, n_legs_in, n_legs_out, phase=0, name=None):
        dom, cod = PRO(n_legs_in), PRO(n_legs_out)
        super().__init__(name, dom, cod, data=phase)
        self.draw_as_spider, self.drawing_name = True, phase or ""
        self.tikzstyle_name = name

    @property
    def name(self):
        return "{}({}, {}{})".format(
            self._name, len(self.dom), len(self.cod),
            ", {}".format(format_number(self.phase)) if self.phase else "")

    def __repr__(self):
        return self.name

    @property
    def phase(self):
        """ Phase of a spider. """
        return self.data

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), -self.phase)

    def subs(self, *args):
        data = cat.rsubs(self.data, *args)
        return type(self)(len(self.dom), len(self.cod), phase=data)

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return Scalar(pi * gradient)\
            @ type(self)(len(self.dom), len(self.cod), self.phase + .5)


class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Z')
        self.color = "green"


class Y(Spider):
    """ Y spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Y')
        self.color = "blue"


class X(Spider):
    """ X spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='X')
        self.color = "red"


class Had(Box):
    """ Hadamard box. """
    def __init__(self):
        super().__init__('H', PRO(1), PRO(1))
        self.draw_as_spider = True
        self.drawing_name, self.tikzstyle_name, = '', 'H'
        self.color, self.shape = "yellow", "rectangle"

    def __repr__(self):
        return self.name

    def dagger(self):
        return self


H = Had()


class Scalar(Box):
    """ Scalar in a ZX diagram. """
    def __init__(self, data):
        super().__init__("scalar", PRO(0), PRO(0), data=data)
        self.drawing_name = format_number(data)

    @property
    def name(self):
        return "scalar({})".format(format_number(self.data))

    def __repr__(self):
        return self.name

    def subs(self, *args):
        data = cat.rsubs(self.data, *args)
        return Scalar(data)

    def dagger(self):
        return Scalar(self.data.conjugate())

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.data.diff(var))


def scalar(data):
    """ Returns a scalar. """
    return Scalar(data)


def gate2zx(box):
    """ Turns gates into ZX diagrams. """
    if isinstance(box, (Bra, Ket)):
        dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
        spiders = [X(dom, cod, phase=.5 * bit) for bit in box.bitstring]
        return Id(0).tensor(*spiders) @ scalar(pow(2, -len(box.bitstring) / 2))
    if isinstance(box, (Rz, Rx)):
        return (Z if isinstance(box, Rz) else X)(1, 1, box.phase)
    if isinstance(box, Controlled) and box.name.startswith('CRz'):
        return Z(1, 2) @ Z(1, 2, box.phase)\
            >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase)) @ Id(1)
    if isinstance(box, Controlled) and box.name.startswith('CRx'):
        return X(1, 2) @ X(1, 2, box.phase)\
            >> Id(1) @ (Z(2, 1) >> X(1, 0, -box.phase)) @ Id(1)
    if isinstance(box, quantum.CU1):
        return Z(1, 2, box.phase) @ Z(1, 2, box.phase)\
            >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase)) @ Id(1)
    if isinstance(box, GatesScalar):
        if box.is_mixed:
            raise NotImplementedError
        return scalar(box.data)
    standard_gates = {
        quantum.H: H,
        quantum.Z: Z(1, 1, .5),
        quantum.X: X(1, 1, .5),
        quantum.Y: Z(1, 1, .5) >> X(1, 1, .5) @ scalar(1j),
        CZ: Z(1, 2) @ Id(1) >> Id(1) @ Had() @ Id(1) >> Id(1) @ Z(2, 1),
        CX: Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)}
    return standard_gates[box]


circuit2zx = Functor(
    ob={qubit: PRO(1)}, ar=gate2zx,
    ob_factory=PRO, ar_factory=Diagram)
