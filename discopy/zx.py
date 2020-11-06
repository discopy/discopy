# -*- coding: utf-8 -*-

""" Implements ZX diagrams. """

from discopy import messages, monoidal, rigid
from discopy.rigid import Box, Diagram, Functor, PRO
from discopy.quantum import Circuit, format_number

from discopy.quantum import Bra, Ket, Rz, Rx, H, CX, CZ, CRz, CRx
from discopy.quantum import Z as PauliZ
from discopy.quantum import X as PauliX
from discopy.quantum import Y as PauliY


class Diagram(rigid.Diagram):
    """ ZX Diagram. """
    def __repr__(self):
        return super().__repr__().replace('Diagram', 'zx.Diagram')

    @staticmethod
    def upgrade(old):
        return Diagram(old.dom, old.cod, old.boxes, old.offsets, old.layers)

    @staticmethod
    def id(dom):
        return Id(len(dom))

    @staticmethod
    def sum(terms, dom=None, cod=None):
        return Sum(terms, dom, cod)

    @staticmethod
    def swap(left, right):
        left = left if isinstance(left, PRO) else PRO(left)
        right = right if isinstance(right, PRO) else PRO(right)
        return monoidal.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)

    @staticmethod
    def permutation(perm, dom=None):
        dom = PRO(len(perm)) if dom is None else dom
        return monoidal.permutation(perm, dom, ar_factory=Diagram)

    @staticmethod
    def cups(left, right):
        return rigid.cups(
            left, right, ar_factory=Diagram, cup_factory=lambda *_: Z(2, 0))

    @staticmethod
    def caps(left, right):
        return rigid.caps(
            left, right, ar_factory=Diagram, cap_factory=lambda *_: Z(0, 2))

    def draw(self, **params):
        return super().draw(**dict(params, draw_types=False))

    def grad(self, var):
        """
        Gradient with respect to `var`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

        Returns
        -------
        diagrams : Sum

        Examples
        --------
        >>> from sympy.abc import phi
        >>> assert Z(1, 1, phi).grad(phi) == scalar(0.5j) @ Z(1, 1, phi - 1)
        """
        return Circuit.grad(self, var)

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
        >>> assert graph.type(4) == graph.type(5) == VertexType.X
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
        from pyzx import Graph, VertexType
        graph, scan = Graph(), []
        for i, _ in enumerate(self.dom):
            scan.append(graph.add_vertex(VertexType.BOUNDARY))
            graph.set_position(scan[-1], i, 0)
        graph.inputs = scan
        for row, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box, Spider):
                node = graph.add_vertex(
                    VertexType.Z if isinstance(box, Z) else VertexType.X,
                    phase=box.phase * 2 if box.phase else None)
                graph.set_position(node, offset, row + 1)
                for i, _ in enumerate(box.dom):
                    graph.add_edge((scan[offset + i], node))
                scan = scan[:offset] + len(box.cod) * [node]\
                    + scan[offset + len(box.dom):]
            if isinstance(box, Swap):
                scan = scan[:offset] + [scan[offset + 1], scan[offset]]\
                    + scan[offset + 2:]
        for i, _ in enumerate(self.cod):
            node = graph.add_vertex(VertexType.BOUNDARY)
            graph.add_edge((scan[i], node))
            graph.set_position(node, i, len(self) + 1)
            scan[i] = node
        graph.outputs = scan
        return graph

    @staticmethod
    def from_pyzx(graph):
        """
        Takes a :class:`pyzx.Graph` returns a :class:`zx.Diagram`.

        >>> bialgebra = Z(1, 2, .25) @ Z(1, 2, .75)\\
        ...     >> Id(1) @ SWAP @ Id(1) >> X(2, 1, .5) @ X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert Diagram.from_pyzx(graph) == bialgebra
        """
        def node2box(node, n_legs_in, n_legs_out):
            from pyzx import VertexType
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return (Z if graph.type(node) == VertexType.Z else X)(
                n_legs_in, n_legs_out, graph.phase(node) * .5)

        def move(scan, source, target):
            swaps = Id(target)\
                @ Diagram.swap(source - target, 1)\
                @ Id(len(scan) - source - 1)
            scan = scan[:target] + [node]\
                + scan[target:source] + scan[source + 1:]
            return scan, swaps

        def make_wires_adjacent(scan, diagram, inputs):
            if not inputs:
                return scan, diagram, len(scan)
            i0 = scan.index(inputs[0])
            for i, node in enumerate(inputs[1:]):
                source, target = scan.index(inputs[i + 1]), i0 + i + 1
                if source != target:
                    scan, swaps = move(scan, source, target)
                    diagram = diagram >> swaps
            return scan, diagram, i0

        diagram, scan = Id(len(graph.inputs)), graph.inputs
        for node in [v for v in graph.vertices()
                 if v not in graph.inputs + graph.outputs]:
            inputs = [v for v in graph.neighbors(node) if v < node]
            outputs = [v for v in graph.neighbors(node) if v > node]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            diagram = diagram >> Id(offset)\
                @ node2box(node, len(inputs), len(outputs))\
                @ Id(len(diagram.cod) - offset - len(inputs))
            scan = scan[:offset] + len(outputs) * [node]\
                + scan[offset + len(inputs):]
        for target, output in enumerate(graph.outputs):
            node, = graph.neighbors(output)
            scan, swaps = move(scan, scan.index(node), target)
            diagram = diagram >> swaps
        return diagram


class Id(rigid.Id, Diagram):
    """ Identity ZX diagram. """
    def __init__(self, dom):
        super().__init__(PRO(dom))

    def __repr__(self):
        return "Id({})".format(len(self.dom))

    __str__ = __repr__


class Sum(monoidal.Sum, Diagram):
    """ Sum of ZX diagrams. """
    @staticmethod
    def upgrade(old):
        return Sum(old.terms, old.dom, old.cod)


class Box(rigid.Box, Diagram):
    """ Box in a ZX diagram. """
    def __init__(self, name, dom, cod, data=None):
        if not isinstance(dom, PRO):
            raise TypeError(messages.type_err(PRO, dom))
        if not isinstance(cod, PRO):
            raise TypeError(messages.type_err(PRO, cod))
        rigid.Box.__init__(self, name, dom, cod, data)
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

    def subs(self, var, expr):
        return type(self)(len(self.dom), len(self.cod),
                          phase=super().subs(var, expr).data)

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return Scalar(.5j * gradient)\
            @ type(self)(len(self.dom), len(self.cod), self.phase - 1)


class Z(Spider):
    """ Z spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Z')
        self.color = "green"


class Y(Spider):
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Y')
        self.color = "blue"


class X(Spider):
    """ X spider. """
    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, phase, name='Y')
        self.color = "red"


class Had(Box, Diagram):
    def __init__(self):
        name = "Had()"
        dom, cod = PRO(1), PRO(1)
        Box.__init__(self, name, dom, cod)
        Diagram.__init__(self, dom, cod, [self], [0])
        self.draw_as_spider = True
        self.color = "yellow"

    @property
    def name(self):
        return self.data or "H"

    def __str__(self):
        return self._name

    def __repr__(self):
        return str(self)


class Scalar(Box):
    """ Scalar in a ZX diagram. """
    def __init__(self, data):
        super().__init__("zx.scalar", PRO(0), PRO(0), data)

    @property
    def name(self):
        return "zx.scalar({})".format(format_number(self.data))

    def __repr__(self):
        return self.name

    def subs(self, var, expr):
        return Scalar(super().subs(var, expr).data)

    def dagger(self):
        return Scalar(self.data.conjugate())

    def grad(self, var):
        if var not in self.free_symbols:
            return Sum([], self.dom, self.cod)
        return Scalar(self.data.diff(var))


def scalar(data):
    """ Returns a scalar. """
    return Scalar(data)


def box2zx(box):
    from functools import reduce
    if isinstance(box, Bra):
        bits = box.bitstring
        return reduce(lambda a, b: a @ b, map(lambda p: X(1, 0, phase=p), bits))
    elif isinstance(box, Ket):
        bits = box.bitstring
        return reduce(lambda a, b: a @ b, map(lambda p: X(0, 1, phase=p), bits))
    elif isinstance(box, Rz):
        return Z(1,1, box.phase)
    elif isinstance(box, Rx):
        return X(1,1, box.phase)
    elif box == H:
        return Had()
    elif box == CX:
        return Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)
    elif box == CZ:
        return Z(1, 2) @ Id(1) >> Id(1) @ Had() @ Id(1) >> Id(1) @ Z(2, 1)
    elif box == PauliZ:
        return Z(1,1,1)
    elif box == PauliY:
        return Y(1,1,1)
    elif box == PauliX:
        return X(1,1,1)
    elif isinstance(box, CRz):
        p = box.phase
        return Z(1, 2) @ Z(1, 2, p) >> Id(1) @ (X(2, 1) >> Z(1, 0, -p)) @ Id(1)
    elif isinstance(box, CRx):
        p = box.phase
        return X(1, 2) @ X(1, 2, p) >> Id(1) @ (Z(2, 1) >> X(1, 0, -p)) @ Id(1)
    elif isinstance(box, CU1):
        p = box.phase
        return Z(1, 2, p) @ Z(1, 2, p) >> Id(1) @ (X(2, 1) >> Z(1, 0, -p)) @ Id(1)


    return box

circuit2zx = Functor(
    ob=lambda x: x, ar=box2zx,
    ob_factory=rigid.Ty, ar_factory=rigid.Diagram)