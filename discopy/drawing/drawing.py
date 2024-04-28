"""
The category of labeled progressive plane graphs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Point
    PlaneGraph
    Drawing
    Equation
"""


from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

import networkx as nx

from discopy.drawing import backend, Node
from discopy.config import DRAWING_ATTRIBUTES
from discopy.utils import (
    Composable, Whiskerable, assert_isinstance, assert_iscomposable, unbiased)

if TYPE_CHECKING:
    from discopy import monoidal


class Point(NamedTuple):
    """ A point is a pair of floats for the x and y coordinates. """
    x: float
    y: float

    def shift(self, x=0, y=0):
        return Point(self.x + x, self.y + y)


class PlaneGraph(NamedTuple):
    """ A plane graph is a graph with a mapping from nodes to points. """
    graph: nx.DiGraph
    positions: dict[Node, Point]


@dataclass
class Drawing(Composable, Whiskerable):
    """
    A drawing is a plane graph with designated input and output types.

    Parameters:
        inside (PlaneGraph) : The plane graph underlying the drawing.
        dom (monoidal.Ty) : The domain of the drawing, i.e. its input type.
        cod (monoidal.Ty) : The codomain of the drawing, i.e. its output type.
        boxes (tuple[monoidal.Box, ...]) : The boxes inside the drawing.
        width (float) : The width of the drawing.
        height (float) : The height of the drawing.
        _check (bool) : Whether to call :meth:`validate_attributes`.

    .. admonition:: Summary

        .. autosummary::

            validate_attributes
            draw
            from_box
            id
            then
            tensor
            dagger
            bubble
            frame
    """
    inside: PlaneGraph
    dom: "monoidal.Ty"
    cod: "monoidal.Ty"
    boxes: tuple["monoidal.Box", ...] = ()
    width: float = 0
    height: float = 0

    graph = property(lambda self: self.inside.graph)
    nodes = property(lambda self: self.graph.nodes)
    edges = property(lambda self: self.graph.edges)
    positions = property(lambda self: self.inside.positions)

    def __init__(
            self, inside, dom, cod, boxes=(), width=0, height=0, _check=True):
        self.inside, self.dom, self.cod = inside, dom, cod
        self.boxes, self.width, self.height = boxes, width, height
        if _check:
            self.validate_attributes()

    def validate_attributes(self):
        """
        Check that the attributes of a drawing are consistent.

        >>> from discopy.monoidal import Ty, Id
        >>> x = Ty('x')
        >>> drawing = Id(x).to_drawing()
        >>> drawing.add_edges([(Node("cod", i=0, x=x), Node("dom", i=0, x=x))])
        >>> drawing.validate_attributes()
        Traceback (most recent call last):
        ...
        ValueError: Wrong edge Node('cod', i=0, x=x) -> Node('dom', i=0, x=x)
        """
        from discopy.monoidal import Ty, Box
        assert_isinstance(self.dom, Ty)
        assert_isinstance(self.cod, Ty)
        for box in self.boxes:
            assert_isinstance(box, Box)
        assert self.dom_nodes == [
            Node("dom", i=i, x=x) for i, x in enumerate(self.dom)]
        assert self.cod_nodes == [
            Node("cod", i=i, x=x) for i, x in enumerate(self.cod)]
        assert self.box_nodes == [
            Node("box", j=j, box=box) for j, box in enumerate(self.boxes)]
        for j, box in enumerate(self.boxes):
            box_node = self.box_nodes[j]
            box_dom_nodes, box_cod_nodes = ([
                Node(f"box_{kind}", i=i, j=j, x=x)
                for i, x in enumerate(xs)] for kind, xs in [
                    ("dom", box.dom), ("cod", box.cod)])
            assert list(self.graph.predecessors(box_node)) == box_dom_nodes
            assert list(self.graph.successors(box_node)) == box_cod_nodes
        for source, target in self.edges:
            if source.kind == "box":
                assert target.kind == "box_cod"
            elif source.kind == "box_dom":
                assert target.kind == "box"
            elif source.kind in ("dom", "box_cod"):
                assert target.kind in ("cod", "box_dom")
            else:
                raise ValueError(f"Wrong edge {source} -> {target}")

        assert self.height >= (1 if self.boxes else 0)
        assert self.width >= (1 if self.boxes else 0)
        assert self.width >= max(x for (x, _) in self.positions.values())
        assert self.height >= max(y for (_, y) in self.positions.values())

        assert set(self.positions.keys()) == set(self.nodes) == set(
            self.dom_nodes + self.cod_nodes) + set(
                self.box_dom_nodes + self.box_nodes + self.box_cod_nodes)
        assert all(isinstance(x, Point) for x in self.positions.values())

    def __eq__(self, other):
        if not isinstance(other, Drawing):
            return False
        return self.is_parallel(other) and self.positions == other.positions

    def draw(self, **params):
        """ Call :meth:`recenter_box_nodes` then :func:`backend.draw`. """
        asymmetry = params.pop("asymmetry", 0.25 * any(
            box.is_dagger
            or getattr(box, "is_conjugate", False)
            or getattr(box, "is_transpose", False)
            for box in self.boxes))
        self.recenter_box_nodes()
        return backend.draw(self, asymmetry=asymmetry, **params)

    def recenter_box_nodes(self):
        """ Recenter boxes in the middle of their input and output wires. """
        for j, box in enumerate(self.boxes):
            if not box.dom and not box.cod:
                continue
            if box.draw_as_wires or box.draw_as_spider:
                if len(box.dom) == 1 or len(box.cod) == 1:
                    continue
            box_dom_nodes, box_cod_nodes = ([
                    Node(kind, i=i, j=j, x=x) for i, x in enumerate(xs)]
                for kind, xs in [("box_dom", box.dom), ("box_cod", box.cod)])
            left, right = (minmax(
                    self.positions[n].x for n in box_dom_nodes + box_cod_nodes)
                for minmax in (min, max))
            box_node = Node("box", j=j, box=box)
            self.positions[box_node] = Point(
                (right + left) / 2, self.positions[box_node].y)

    def union(self, other, dom, cod, _check=True):
        """ Take the union of two drawings, assuming nodes are distinct. """
        graph = nx.union(self.inside.graph, other.inside.graph)
        inside = PlaneGraph(graph, self.positions | other.positions)
        boxes = self.boxes + other.boxes
        width = max(self.width, other.width)
        height = max(self.height, other.height)
        return Drawing(inside, dom, cod, boxes, width, height, _check)

    def add_nodes(self, positions: dict[Node, Point]):
        """ Add nodes to the graph given their positions. """
        if not positions:
            return
        self.graph.add_nodes_from({
            n: dict(box=n.box) if n.kind == "box" else dict(kind=n.kind, i=n.i)
            for n in positions})
        self.positions.update(positions)
        self.width = max(self.width, max(i for (i, _) in positions.values()))
        self.height = max(self.height, max(j for (_, j) in positions.values()))

    def add_edges(self, edges: list[tuple[Node, Node]]):
        """ Add edges from a list. """
        self.graph.add_edges_from(edges)

    @property
    def actual_width(self):
        """
        The difference between max and min x coordinate.
        This can be different from width when dom and cod have length <= 1.
        """
        return max(
                [x for (x, _) in self.positions.values()] + [0]
            ) - min([x for (x, _) in self.positions.values()] + [0])

    def set_width_and_height(self):
        """ Compute the width and height and update the attributes. """
        self.width = max([x for (x, _) in self.positions.values()] + [0])
        self.height = max([y for (_, y) in self.positions.values()] + [0])
        return self

    def relabel_nodes(
            self, mapping=dict(), positions=dict(), copy=True, _check=False):
        """ Relabel nodes and/or their positions. """
        graph = nx.relabel_nodes(self.graph, mapping, copy)
        if copy:
            positions = {mapping.get(node, node): positions.get(node, pos)
                         for node, pos in self.positions.items()}
            inside, boxes = PlaneGraph(graph, positions), self.boxes
            result = Drawing(inside, self.dom, self.cod, boxes, _check=_check)
            return result.set_width_and_height()
        self.positions.update(positions)
        return self.set_width_and_height()

    def make_space(self, space, x,
                   y_min=None, y_max=None, inclusive=True, copy=False):
        """
        Make some horizontal space after position x
        for all nodes between y_min and y_max (inclusive).

        Example
        -------
        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Drawing.from_box(Box('f', x @ y, z))
        >>> f.make_space(2, 0.5, 0.75, 1.0, copy=True).draw(
        ...     aspect='equal', path="docs/_static/drawing/make-space.png")

        .. image:: /_static/drawing/make-space.png
            :align: center
        """
        y_min = 0 if y_min is None else y_min
        y_max = self.height if y_max is None else y_max
        return self.relabel_nodes(copy=copy, positions={
            n: p.shift(x=((p.x >= x) * space))
            for n, p in self.positions.items()
            if (y_min <= p.y <= y_max if inclusive else y_min < p.y < y_max)})

    @property
    def is_identity(self):
        """ A drawing with no boxes is the identity. """
        return not self.boxes

    @property
    def is_empty(self):
        """ A drawing with no boxes and no wires is empty. """
        return self.is_identity and not self.dom

    @property
    def is_box(self):
        """ Whether the drawing is just one box. """
        return len(self.boxes) == 1 and self.is_parallel(self.boxes[0])

    @property
    def box(self):
        """ Syntactic sugar for self.boxes[0] when self.is_box """
        if not self.is_box:
            raise ValueError
        return self.boxes[0]

    @staticmethod
    def from_box(box: "monoidal.Box") -> Drawing:
        """
        Draw a diagram with just one box.

        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ z)
        >>> assert f.to_drawing() == Drawing.from_box(f)
        >>> for ps in f.to_drawing().positions.items(): print(*ps)
        Node('box', box=f, j=0) Point(x=0.5, y=0.5)
        Node('dom', i=0, x=x) Point(x=0.5, y=1)
        Node('box_dom', i=0, j=0, x=x) Point(x=0.5, y=0.75)
        Node('box_cod', i=0, j=0, x=y) Point(x=0.0, y=0.25)
        Node('box_cod', i=1, j=0, x=z) Point(x=1.0, y=0.25)
        Node('cod', i=0, x=y) Point(x=0.0, y=0)
        Node('cod', i=1, x=z) Point(x=1.0, y=0)
        """
        from discopy.monoidal import Box
        box_dom, box_cod = box.dom.to_drawing(), box.cod.to_drawing()
        old_box, box = box, Box(
            box.name, box_dom, box_cod, is_dagger=box.is_dagger)

        for attr, default in DRAWING_ATTRIBUTES.items():
            setattr(box, attr, getattr(old_box, attr, default(box)))

        bubble_opening, bubble_closing = box.bubble_opening, box.bubble_closing

        if box.draw_as_wires and not (bubble_opening or bubble_closing):
            for obj in box.cod.inside:
                obj.reposition_label = True

        if bubble_opening:
            width = max(1, len(box.dom), len(box.cod[1:-1]))
        elif bubble_closing:
            width = max(1, len(box.dom[1:-1]), len(box.cod))
        else:
            width = max(1, len(box.dom) - 1, len(box.cod) - 1)

        inside = PlaneGraph(nx.DiGraph(), dict())
        result = Drawing(
            inside, box.dom, box.cod, (box, ), width, 1, _check=False)

        box_node = Node("box", box=box, j=0)
        result.add_nodes({box_node: Point(width / 2, 0.5)})

        dom = [Node("dom", i=i, x=x) for i, x in enumerate(box.dom)]
        cod = [Node("cod", i=i, x=x) for i, x in enumerate(box.cod)]
        box_dom = [Node("box_dom", i=i, j=0, x=x.x) for i, x in enumerate(dom)]
        box_cod = [Node("box_cod", i=i, j=0, x=x.x) for i, x in enumerate(cod)]

        result.add_edges(list(zip(dom, box_dom)))
        result.add_edges([(x, box_node) for x in box_dom])
        result.add_edges([(box_node, x) for x in box_cod])
        result.add_edges(list(zip(box_cod, cod)))

        if bubble_opening:
            result.add_nodes({
                cod[0]: Point(0, 0), box_cod[0]: Point(0, 0),
                cod[-1]: Point(width, 0), box_cod[-1]: Point(width, 0)})
            cod, box_cod = cod[1:-1], box_cod[1:-1]
        if bubble_closing:
            result.add_nodes({
                dom[0]: Point(0, 1), box_dom[0]: Point(0, 1),
                dom[-1]: Point(width, 1), box_dom[-1]: Point(width, 1)})
            dom, box_dom = dom[1:-1], box_dom[1:-1]
        result.add_nodes({
            x: Point(i + (width - len(xs) + 1) / 2, y) for xs, y in [
                (dom, 1),
                (box_dom, 1 if box.draw_as_wires else 0.75),
                (box_cod, 0 if box.draw_as_wires else 0.25),
                (cod, 0)]
            for i, x in enumerate(xs)})
        return result

    @staticmethod
    def id(dom: "monoidal.Ty", length=0) -> Drawing:
        """ Draw the identity diagram. """
        inside = PlaneGraph(nx.DiGraph(), dict())
        result = Drawing(inside, dom, dom, width=len(dom), _check=False)
        dom_nodes = [Node("dom", i=i, x=x) for i, x in enumerate(dom)]
        cod_nodes = [Node("cod", i=i, x=x) for i, x in enumerate(dom)]
        offset = 0 if len(dom) > 1 else 0.5
        result.add_nodes({
            x: Point(i + offset, 1) for i, x in enumerate(dom_nodes)})
        result.add_nodes({
            x: Point(i + offset, 0) for i, x in enumerate(cod_nodes)})
        result.add_edges(list(zip(dom_nodes, cod_nodes)))
        return result

    @unbiased
    def then(self, other: Drawing) -> Drawing:
        """
        Draw one diagram composed with another.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Drawing.from_box(Box('f', x, y @ z))
        >>> g = Drawing.from_box(Box('g', z @ y, x))
        >>> d = f @ y >> y @ g
        >>> d.draw(path="docs/_static/drawing/composition.png")

        .. image:: /_static/drawing/composition.png
            :align: center
        """
        assert_iscomposable(self, other)
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        dom, cod = self.dom, other.cod
        tmp_cod = [Node("tmp_cod", i=i) for i, n in enumerate(self.cod_nodes)]
        tmp_dom = [Node("tmp_dom", i=i) for i, n in enumerate(other.dom_nodes)]
        mapping = {
            n: n.shift_j(len(self.boxes)) for n in other.box_nodes + (
                other.box_dom_nodes + other.box_cod_nodes)}
        mapping.update(dict(zip(other.dom_nodes, tmp_dom)))
        positions = {
            n: p.shift(y=other.height + 1) for n, p in self.positions.items()}
        result = self.relabel_nodes(
            dict(zip(self.cod_nodes, tmp_cod)), positions, _check=False).union(
                other.relabel_nodes(mapping), dom, cod, _check=False)
        cut = other.height + 0.5
        for i, (u, v) in enumerate(zip(self.cod_nodes, other.dom_nodes)):
            top = result.positions[tmp_cod[i]].x
            bot = result.positions[tmp_dom[i]].x
            if top > bot:
                result.make_space(top - bot, (i > 0) * bot, 0, cut)
            elif top < bot:
                result.make_space(bot - top, (i > 0) * top, cut, result.height)
            source, = self.graph.predecessors(u)
            target, = other.graph.successors(v)
            result.add_edges([(source, mapping.get(target, target))])
        result.graph.remove_nodes_from(tmp_dom + tmp_cod)
        [result.positions.pop(n) for n in tmp_dom + tmp_cod]
        result = result.relabel_nodes(positions={
            n: p.shift(y=-1)
            for n, p in result.positions.items() if p.y > other.height})
        if result.height != self.height + other.height - (
                1 if self.height + other.height > 1 else 0):
            pass
        return result

    def stretch(self, y):
        """
        Stretch input and output wires to increase the height of a diagram
        by a given length.
        .
        Example
        -------
        >>> from discopy.monoidal import Box
        >>> f = Drawing.from_box(Box('f', 'x', 'x'))
        >>> f.stretch(2).draw(path="docs/_static/drawing/stretch.png")

        .. image:: /_static/drawing/stretch.png
            :align: center
        """
        return self.relabel_nodes(positions={n: p.shift(y=(
                y if n.kind == "dom" else 0 if n.kind == "cod" else y / 2))
            for n, p in self.positions.items()})

    @unbiased
    def tensor(self, other: Drawing) -> Drawing:
        """
        Draw two diagrams side by side.

        Example
        -------
        >>> from discopy.monoidal import Box
        >>> f = Drawing.from_box(Box('f', 'x', 'x'))
        >>> d = (f >> f >> f) @ (f >> f)
        >>> d.draw(path="docs/_static/drawing/tensor.png")

        .. image:: /_static/drawing/tensor.png
            :align: center
        """
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        mapping = {
            n: n.shift_j(len(self.boxes)) for n in other.box_nodes + (
                other.box_dom_nodes + other.box_cod_nodes)}
        mapping.update({n: n.shift_i(len(self.dom)) for n in other.dom_nodes})
        mapping.update({n: n.shift_i(len(self.cod)) for n in other.cod_nodes})
        if self.height < other.height:
            self = self.stretch(other.height - self.height)
        elif self.height > other.height:
            other = other.stretch(self.height - other.height)
        x_shift = max(
                [p.x + 1 for p in self.positions.values()] + [0]
            ) - min([p.x for p in other.positions.values()] + [0.5])
        return self.union(other.relabel_nodes(mapping, positions={
            n: p.shift(x=x_shift) for n, p in other.positions.items()}),
            dom=self.dom @ other.dom, cod=self.cod @ other.cod, _check=False)

    def dagger(self) -> Drawing:
        """ The reflection of a drawing along the the horizontal axis. """
        def box_dagger(box):
            result = box.dagger()
            for attr in DRAWING_ATTRIBUTES:
                setattr(result, attr, getattr(box, attr))
            return result

        if self.is_box:
            return Drawing.from_box(box_dagger(self.box))

        mapping = {
            n: Node("box", box=box_dagger(n.box), j=len(self.boxes) - n.j - 1)
            for n in self.nodes if n.kind == "box"}
        mapping.update({
            n: Node(kd, i=n.i, j=len(self.boxes) - n.j - 1, x=n.x)
            for (k, kd) in [("box_dom", "box_cod"), ("box_cod", "box_dom")]
            for n in self.nodes if n.kind == k})
        mapping.update({
            n: Node("cod", i=i, x=n.x) for i, n in enumerate(self.dom_nodes)})
        mapping.update({
            n: Node("dom", i=i, x=n.x) for i, n in enumerate(self.cod_nodes)})
        graph = nx.relabel_nodes(self.graph, mapping).reverse(copy=False)
        inside = PlaneGraph(graph, positions={
            mapping[n]: Point(x, self.height - y)
            for n, (x, y) in self.positions.items()})
        dom, cod = self.cod, self.dom
        boxes = tuple(map(box_dagger, self.boxes[::-1]))
        return Drawing(inside, dom, cod, boxes, self.width, self.height)

    @staticmethod
    def bubble_opening(dom, arg_dom, left, right):
        """
        Construct the opening of a bubble as a box drawn as wires.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.bubble_opening(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/bubble-opening.png")

        .. image:: /_static/drawing/bubble-opening.png
            :align: center
        """
        from discopy.monoidal import Box
        return Drawing.from_box(
            Box("top", dom, left @ arg_dom @ right, bubble_opening=True))

    @staticmethod
    def bubble_closing(arg_cod, cod, left, right):
        """ Construct the closing of a bubble as a drawing with one box. """
        from discopy.monoidal import Box
        return Drawing.from_box(
            Box("bot", left @ arg_cod @ right, cod, bubble_closing=True))

    @staticmethod
    def frame_opening(dom, arg_dom, left, right):
        """
        Construct the opening of a frame as the opening of a bubble squashed to
        zero height so that it looks like half of a rectangle.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.frame_opening(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/frame-opening.png")

        .. image:: /_static/drawing/frame-opening.png
            :align: center
        """
        result = Drawing.bubble_opening(dom, arg_dom, left, right)
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-.5) for n in result.box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=.5) for n in result.box_cod_nodes})
        arg_cod_nodes = result.box_cod_nodes[1:-1]
        result.graph.remove_edges_from([
            (u, v) for u in result.box_dom_nodes for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in arg_cod_nodes])
        return result

    @staticmethod
    def frame_closing(arg_cod, cod, left, right):
        """ Closing of a frame, see :meth:`frame_opening`. """
        result = Drawing.bubble_closing(arg_cod, cod, left, right)
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-.5) for n in result.box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=.5) for n in result.box_cod_nodes})
        arg_dom_nodes = result.box_dom_nodes[1:-1]
        result.graph.remove_edges_from([
            (u, v) for u in arg_dom_nodes for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in result.box_cod_nodes])
        return result

    def bubble(self, dom=None, cod=None, name=None,
               width=None, height=None, draw_as_square=False) -> Drawing:
        """
        Draw a closed line around a drawing, with some wires coming in and out.

        Parameters:
            dom (monoidal.Ty) : The wires coming into the bubble.
            cod (monoidal.Ty) : The wires coming out of the bubble.
            name (str) : The label of the bubble, drawn on the top left.
            width

        >>> from discopy.symmetric import *
        >>> a, b, c, d = map(Ty, "abcd")
        >>> f = Box('f', a @ b, c @ d).to_drawing()
        >>> f.bubble(d @ c @ c, b @ a @ a, name="g").draw(
        ...     path="docs/_static/drawing/bubble-drawing.png")

        .. image:: /_static/drawing/bubble-drawing.png
            :align: center
        """
        dom = self.dom if dom is None else dom
        cod = self.cod if cod is None else cod
        arg_dom, arg_cod = self.dom, self.cod
        left, right = type(dom)(name or ""), type(dom)("")
        left[0].always_draw_label = True
        wires_can_go_straight = (
            len(dom), len(cod)) == (len(arg_dom), len(arg_cod))
        if draw_as_square or not wires_can_go_straight:
            top = Drawing.frame_opening(dom, arg_dom, left, right)
            bot = Drawing.frame_closing(arg_cod, cod, left, right)
        else:
            top = Drawing.bubble_opening(dom, arg_dom, left, right)
            bot = Drawing.bubble_closing(arg_cod, cod, left, right)
        middle = self if height is None else self.stretch(height - self.height)
        result = top >> left @ middle @ right >> bot
        result.relabel_nodes(copy=False, positions={
            n: p.shift(x=-0.5) for n, p in result.positions.items()})
        if width is not None and result.width != width:
            space = (width - result.width) / 2
            result.make_space(space, 0.01)
            result.make_space(space, result.width)
        if len(dom) == len(arg_dom):
            dom_nodes, arg_dom_nodes = ([
                    Node(kind, x=x, i=i + off, j=0)
                    for i, x in enumerate(xs)]
                for (kind, xs, off) in [
                    ("box_dom", dom, 0), ("box_cod", arg_dom, 1)])
            node = Node("box", j=0, box=top.box)
            result.graph.add_edges_from(zip(dom_nodes, arg_dom_nodes))
            result.graph.remove_edges_from([(x, node) for x in dom_nodes])
            result.graph.remove_edges_from([(node, x) for x in arg_dom_nodes])
        if len(cod) == len(arg_cod):
            arg_cod_nodes, cod_nodes = ([
                    Node(kind, x=x, i=i + off, j=len(result.boxes) - 1)
                    for i, x in enumerate(xs)]
                for (kind, xs, off) in [
                    ("box_dom", arg_cod, 1), ("box_cod", cod, 0)])
            node = Node("box", j=len(result.boxes) - 1, box=bot.box)
            result.graph.add_edges_from(zip(arg_cod_nodes, cod_nodes))
            result.graph.remove_edges_from([(x, node) for x in arg_cod_nodes])
            result.graph.remove_edges_from([(node, x) for x in cod_nodes])
        return result

    def frame(self, *others: Drawing,
              dom=None, cod=None, name=None, draw_vertically=False) -> Drawing:
        """
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g, h = Box('f', x, y ** 3), Box('g', y, y @ y), Box('h', x, y)
        >>> f.bubble(dom=x @ x, cod=y @ y, name="b", draw_as_frame=True
        ...     ).draw(path="docs/_static/drawing/single-frame.png")

        .. image:: /_static/drawing/single-frame.png
            :align: center

        >>> Bubble(f, g, h >> h[::-1], dom=x, cod=y @ y
        ...     ).draw(path="docs/_static/drawing/horizontal-frame.png")

        .. image:: /_static/drawing/horizontal-frame.png
            :align: center

        >>> Bubble(f, g, h, dom=x, cod=y @ y, draw_vertically=True
        ...     ).draw(path="docs/_static/drawing/vertical-frame.png")

        .. image:: /_static/drawing/vertical-frame.png
            :align: center
        """
        args, empty = (self, ) + others, type(self.dom)()
        method = "then" if draw_vertically else "tensor"
        params = dict(
                width=max([arg.actual_width for arg in args] + [0]) + 2
            ) if draw_vertically else dict(
                height=max([arg.height for arg in args] + [0]))
        result =  getattr(Drawing.id(empty), method)(*(arg.bubble(
            empty, empty, draw_as_square=True, **params)
            for arg in args)).bubble(dom, cod, name, draw_as_square=True)
        for i, source in enumerate(result.dom_nodes):
            target, = result.graph.successors(source)
            for n in (source, target):
                x = i + (result.width - len(result.dom) + 1) / 2
                result.positions[n] = Point(x, result.positions[n].y)
        for i, target in enumerate(result.cod_nodes):
            source, = result.graph.predecessors(target)
            for n in (source, target):
                x = i + (result.width - len(result.cod) + 1) / 2
                result.positions[n] = Point(x, result.positions[n].y)
        result.relabel_nodes(copy=False, positions={
            n: p.shift(y=0.5)
            for n in result.nodes for p in [result.positions[n]]
            if n.kind == "dom" or "box" in n.kind and n.j == 0})
        return result

    def zero(dom, cod):
        from discopy.monoidal import Box
        result = Box("zero", dom, cod).to_drawing()
        result.zero_drawing = True
        return result

    def add(self, other: Drawing, symbol="+", space=1):
        """ Concatenate two drawings with a symbol in between. """
        from discopy.monoidal import Box
        if getattr(self, "zero_drawing", False):
            return other
        if getattr(other, "zero_drawing", False):
            return self
        empty = type(self.dom)()
        scalar = Box(symbol, empty, empty, draw_as_spider=True, color="white")
        result = self @ scalar.to_drawing() @ other
        result.make_space(space - 1, self.width + 1)  # Right of the scalar.
        result.make_space(space - 1, self.width)  # Left of the scalar.
        return result

    __add__ = add

    def to_drawing(self):
        return self


class Equation:
    """
    An equation is a list of diagrams with a dedicated draw method.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between the terms.
        space : The space between the terms.

    Example
    -------
    >>> from discopy.tensor import Spider, Swap, Dim, Id
    >>> dim = Dim(2)
    >>> mu, eta = Spider(2, 1, dim), Spider(0, 1, dim)
    >>> delta, upsilon = Spider(1, 2, dim), Spider(1, 0, dim)
    >>> special = Equation(mu >> delta, Id(dim))
    >>> frobenius = Equation(
    ...     delta @ Id(dim) >> Id(dim) @ mu,
    ...     mu >> delta,
    ...     Id(dim) @ delta >> mu @ Id(dim))
    >>> Equation(special, frobenius, symbol=', ').draw(
    ...          aspect='equal', draw_type_labels=False, figsize=(8, 2),
    ...          path='docs/_static/drawing/frobenius-axioms.png')

    .. image:: /_static/drawing/frobenius-axioms.png
        :align: center
    """
    def __init__(self, *terms: "monoidal.Diagram", symbol="=", space=1):
        self.terms, self.symbol, self.space = terms, symbol, space

    def __repr__(self):
        return f"Equation({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f" {self.symbol} ".join(map(str, self.terms))

    def to_drawing(self):
        result = self.terms[0].to_drawing()
        for term in self.terms[1:]:
            result = result.add(term.to_drawing(), self.symbol, self.space)
        return result

    def draw(self, path=None, **params):
        """
        Drawing an equation.

        Parameters:
            path : Where to save the drawing.
            params : Passed to :meth:`discopy.monoidal.Diagram.draw`.
        """
        return self.to_drawing().draw(path=path, **params)

    def __bool__(self):
        return all(term == self.terms[0] for term in self.terms)


for kind in ["dom", "cod", "box", "box_dom", "box_cod"]:
    setattr(Drawing, f"{kind}_nodes", property(lambda self, kind=kind: [
        node for node in self.nodes if node.kind == kind]))
