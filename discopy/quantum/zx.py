# -*- coding: utf-8 -*-

"""
ZX-calculus diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Spider
    Z
    Y
    X
    Scalar
"""

from math import pi

from discopy import cat, rigid, tensor, quantum
from discopy.cat import factory
from discopy.frobenius import Category
from discopy.quantum.circuit import qubit, Circuit
from discopy.quantum.gates import (
    Bra, Ket, Rz, Rx, CX, CZ, Controlled, format_number)
from discopy.quantum.gates import Scalar as GatesScalar
from discopy.rigid import Sum, PRO
from discopy.utils import factory_name


@factory
class Diagram(tensor.Diagram[complex]):
    """ ZX Diagram. """
    ty_factory = PRO

    @staticmethod
    def swap(left, right):
        left = left if isinstance(left, PRO) else PRO(left)
        right = right if isinstance(right, PRO) else PRO(right)
        return tensor.Diagram.swap.__func__(Diagram, left, right)

    @staticmethod
    def permutation(perm, dom=None):
        dom = PRO(len(perm)) if dom is None else dom
        return tensor.Diagram.permutation.__func__(Diagram, perm, dom)

    @staticmethod
    def cup_factory(left, right):
        del left, right
        return Z(2, 0)

    def grad(self, var, **params) -> rigid.Sum:
        """
        Gradient with respect to `var`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

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
        >>> assert (graph.inputs(), graph.outputs()) == ((0, 1), (6, 7))
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
            graph.set_inputs(graph.inputs() + (node,))
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
                raise NotImplementedError
        for i, _ in enumerate(self.cod):
            target = graph.add_vertex(VertexType.BOUNDARY)
            source, hadamard = scan[i]
            etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
            graph.add_edge((source, target), etype)
            graph.set_position(target, i, len(self) + 1)
            graph.set_outputs(graph.outputs() + (target,))
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
        * a boundary node is not in :code:`graph.inputs() + graph.outputs()`,
        * or :code:`set(graph.inputs()).intersection(graph.outputs())`.
        """
        from pyzx import VertexType, EdgeType

        def node2box(node, n_legs_in, n_legs_out):
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return \
                (Z if graph.type(node) == VertexType.Z else X)(  # noqa: E721
                    n_legs_in, n_legs_out, graph.phase(node) * .5)

        def move(scan, source, target):
            if target < source:
                swaps = Id(target)\
                    @ Diagram.swap(source - target, 1)\
                    @ Id(len(scan) - source - 1)
                scan = scan[:target] + (scan[source],)\
                    + scan[target:source] + scan[source + 1:]
            elif target > source:
                swaps = Id(source)\
                    @ Diagram.swap(1, target - source)\
                    @ Id(len(scan) - target - 1)
                scan = scan[:source] + scan[source + 1:target]\
                    + (scan[source],) + scan[target:]
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
            graph.type(node) == VertexType.BOUNDARY  # noqa: E721
            and node not in graph.inputs() + graph.outputs()
            for node in graph.vertices())
        if missing_boundary:
            raise ValueError
        duplicate_boundary = set(graph.inputs()).intersection(graph.outputs())
        if duplicate_boundary:
            raise ValueError
        diagram, scan = Id(len(graph.inputs())), graph.inputs()
        for node in [v for v in graph.vertices()
                     if v not in graph.inputs() + graph.outputs()]:
            inputs = [v for v in graph.neighbors(node) if v < node
                      and v not in graph.outputs() or v in graph.inputs()]
            inputs.sort(key=scan.index)
            outputs = [v for v in graph.neighbors(node) if v > node
                       and v not in graph.inputs() or v in graph.outputs()]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            hadamards = Id().tensor(*[
                H if graph.edge_type((i, node)) == EdgeType.HADAMARD
                else Id(1) for i in scan[offset: offset + len(inputs)]])
            box = node2box(node, len(inputs), len(outputs))
            diagram = diagram >> Id(offset) @ (hadamards >> box)\
                @ Id(len(diagram.cod) - offset - len(inputs))
            scan = scan[:offset] + len(outputs) * (node,)\
                + scan[offset + len(inputs):]
        for target, output in enumerate(graph.outputs()):
            node, = graph.neighbors(output)
            etype = graph.edge_type((node, output))
            hadamard = H if etype == EdgeType.HADAMARD else Id(1)
            scan, swaps = move(scan, scan.index(node), target)
            diagram = diagram >> swaps\
                >> Id(target) @ hadamard @ Id(len(scan) - target - 1)
        return diagram


class Box(tensor.Box[complex], Diagram):
    """
    A ZX box is a tensor box in a ZX diagram.

    Parameters:
        name (str) : The name of the box.
        dom (rigid.PRO) : The domain of the box, i.e. its input.
        cod (rigid.PRO) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (tensor.Box, )


class Sum(tensor.Sum[complex], Box):
    """
    A formal sum of ZX diagrams with the same domain and codomain.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Dim) : The domain of the formal sum.
        cod (Dim) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (tensor.Sum, )


class Swap(tensor.Swap[complex], Box):
    """ Swap in a ZX diagram. """
    def __repr__(self):
        return "SWAP"

    __str__ = __repr__


class Spider(tensor.Spider[complex], Box):
    """ Abstract spider box. """

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, PRO(1), phase)
        factory_str = type(self).__name__
        phase_str = f", {self.phase}" if self.phase else ""
        self.name = f"{factory_str}({n_legs_in}, {n_legs_out}{phase_str})"

    def __setstate__(self, state):
        if "_name" in state and state["_name"] == type(self).__name__:
            phase = state.get("_data", None)
            phase_str = f', {phase}' if phase else ''
            state["_name"] = (
                type(self).__name__ +
                f"({state['_dom'].n}, {state['_cod'].n}{phase_str})"
            )
        super().__setstate__(state)

    def __repr__(self):
        return str(self).replace(type(self).__name__, factory_name(type(self)))

    def subs(self, *args):
        phase = cat.rsubs(self.phase, *args)
        return type(self)(len(self.dom), len(self.cod), phase=phase)

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return Scalar(pi * gradient)\
            @ type(self)(len(self.dom), len(self.cod), self.phase + .5)

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), -self.phase)

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.phase)

    @property
    def array(self):
        return None


class Z(Spider):
    """ Z spider. """
    tikzstyle_name = 'Z'
    color = 'green'


class Y(Spider):
    """ Y spider. """
    tikzstyle_name = 'Y'
    color = "blue"


class X(Spider):
    """ X spider. """
    tikzstyle_name = 'X'
    color = "red"


class Scalar(Box):
    """ Scalar in a ZX diagram. """
    def __init__(self, data):
        super().__init__("scalar", PRO(0), PRO(0), data=data)
        self.drawing_name = format_number(data)

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def subs(self, *args):
        data = cat.rsubs(self.data, *args)
        return Scalar(data)

    def dagger(self):
        return Scalar(self.data.conjugate())

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.data.diff(var))


def scalar(data):
    """ Returns a scalar. """
    return Scalar(data)


def gate2zx(box):
    """ Turns gates into ZX diagrams. """
    if isinstance(box, (Bra, Ket)):
        dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
        spiders = [X(dom, cod, phase=.5 * bit) for bit in box.bitstring]
        return Id().tensor(*spiders) @ scalar(pow(2, -len(box.bitstring) / 2))
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
    if isinstance(box, Controlled) and box.distance != 1:
        return circuit2zx(box._decompose())
    standard_gates = {
        quantum.H: H,
        quantum.Z: Z(1, 1, .5),
        quantum.X: X(1, 1, .5),
        quantum.Y: Z(1, 1, .5) >> X(1, 1, .5) @ scalar(1j),
        quantum.S: Z(1, 1, .25),
        quantum.T: Z(1, 1, .125),
        CZ: Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1),
        CX: Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1) @ scalar(2 ** 0.5)}
    return standard_gates[box]


circuit2zx = quantum.circuit.Functor(
    ob={qubit: PRO(1)}, ar=gate2zx,
    dom=Category(quantum.circuit.Ty, Circuit), cod=Category(PRO, Diagram))

H = Box('H', PRO(1), PRO(1))
H.dagger = lambda: H
H.draw_as_spider = True
H.drawing_name, H.tikzstyle_name, = '', 'H'
H.color, H.shape = "yellow", "rectangle"

SWAP = Swap(PRO(1), PRO(1))
Diagram.braid_factory, Diagram.sum_factory = Swap, Sum
Id = Diagram.id
