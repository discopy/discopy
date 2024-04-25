"""
The category of labeled progressive plane graphs.

Example
-------
>>> from discopy.drawing import spiral
>>> F = lambda f: f.to_drawing()
>>> d = F(spiral(1)) @ F(spiral(2)) @ F(spiral(3))
>>> d.draw(path="docs/_static/drawing/spiral-tensor.png")

.. image:: /_static/drawing/spiral-tensor.png
    :align: center
"""


from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

import networkx as nx

from discopy.drawing import backend, Node
from discopy.config import DRAWING_ATTRIBUTES
from discopy.utils import (
    Composable, Whiskerable, assert_iscomposable, unbiased)

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
    """ A diagram drawing is a plane graph with designated dom and cod. """
    inside: PlaneGraph
    dom: "monoidal.Ty"
    cod: "monoidal.Ty"
    boxes: tuple["monoidal.Box", ...] = ()
    width: float = 0.
    height: float = 0.

    graph = property(lambda self: self.inside.graph)
    nodes = property(lambda self: self.graph.nodes)
    positions = property(lambda self: self.inside.positions)

    def __eq__(self, other):
        if not isinstance(other, Drawing):
            return False
        return self.is_parallel(other) and self.positions == other.positions

    @property
    def dom_nodes(self):
        return [node for node in self.nodes if node.kind == "dom"]

    @property
    def cod_nodes(self):
        return [node for node in self.nodes if node.kind == "cod"]

    @property
    def box_nodes(self):
        return [node for node in self.nodes if "box" in node.kind]

    def draw(self, **params):
        if not self.height:
            return self.stretch(1).draw(**params)
        return backend.draw(*self.inside, **params)

    def union(self, other, dom, cod):
        graph = nx.union(self.inside.graph, other.inside.graph)
        inside = PlaneGraph(graph, self.positions | other.positions)
        boxes = self.boxes + other.boxes
        width = max(self.width, other.width)
        height = max(self.height, other.height)
        return Drawing(inside, dom, cod, boxes, width, height)

    def add_nodes(self, positions: dict[Node, Point]):
        if not positions:
            return
        self.graph.add_nodes_from({
            n: dict(box=n.box) if n.kind == "box" else dict(kind=n.kind, i=n.i)
            for n in positions})
        self.positions.update(positions)
        self.width = max(self.width, max(i for (i, _) in positions.values()))
        self.height = max(self.height, max(j for (_, j) in positions.values()))

    def add_edges(self, edges: list[tuple[Node, Node]]):
        self.graph.add_edges_from(edges)

    def set_width_and_height(self):
        self.width = max([x for (x, _) in self.positions.values()] + [0])
        self.height = max([y for (_, y) in self.positions.values()] + [0])
        return self

    def relabel_nodes(self, mapping=dict(), positions=dict(), copy=True):
        graph = nx.relabel_nodes(self.graph, mapping, copy)
        if copy:
            positions = {mapping.get(node, node): positions.get(node, pos)
                         for node, pos in self.positions.items()}
            inside, boxes = PlaneGraph(graph, positions), self.boxes
            result = Drawing(inside, self.dom, self.cod, boxes)
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
    def is_box(self):
        """ Whether the drawing is just one box. """
        return len(self.boxes) == 1 and self.is_parallel(self.boxes[0])

    @property
    def box(self):
        if not self.is_box:
            raise ValueError
        return self.boxes[0]

    @staticmethod
    def from_box(box: "monoidal.Box") -> Drawing:
        """ Draw a diagram with just one box. """
        for attr, default in DRAWING_ATTRIBUTES.items():
            setattr(box, attr, getattr(box, attr, default(box)))
        frame_opening, frame_closing = box.frame_opening, box.frame_closing
        box_node = Node("box", box=box, j=0)
        width, height = max(1, len(box.dom), len(box.cod)), 1
        dom = [Node("dom", i=i, x=x) for i, x in enumerate(box.dom.inside)]
        cod = [Node("cod", i=i, x=x) for i, x in enumerate(box.cod.inside)]
        box_dom = [Node("box_dom", i=i, j=0, x=x.x) for i, x in enumerate(dom)]
        box_cod = [Node("box_cod", i=i, j=0, x=x.x) for i, x in enumerate(cod)]

        inside = PlaneGraph(nx.DiGraph(), dict())
        result = Drawing(inside, box.dom, box.cod, (box, ), width, height)

        result.add_nodes({
            x: Point(0.5 + i, y) for xs, y in [
                (dom, 1), (box_dom, 0.75), (box_cod, 0.25), (cod, 0)]
            for i, x in enumerate(xs)})
        result.add_nodes({box_node: Point(width / 2, 0.5)})

        space = (len(dom) - len(cod)) / 2
        if space > 0:
            result.make_space(space, 0, 0, .25)
        elif space < 0:
            result.make_space(-space, 0, .75, 1)

        result.add_edges(list(zip(dom, box_dom)))
        result.add_edges([(x, box_node) for x in box_dom])
        result.add_edges([(box_node, x) for x in box_cod])
        result.add_edges(list(zip(box_cod, cod)))

        return result

    @staticmethod
    def id(dom: "monoidal.Ty", length=0) -> Drawing:
        """ Draw the identity diagram. """
        result = Drawing(PlaneGraph(nx.DiGraph(), dict()), dom, dom)
        dom_nodes = [Node("dom", i=i, x=x) for i, x in enumerate(dom.inside)]
        cod_nodes = [Node("cod", i=i, x=x) for i, x in enumerate(dom.inside)]
        result.add_nodes({x: Point(i, 0) for i, x in enumerate(dom_nodes)})
        result.add_nodes({x: Point(i, 0) for i, x in enumerate(cod_nodes)})
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
        dom, cod = self.dom, other.cod
        tmp_cod = [Node("tmp_cod", i=i) for i, n in enumerate(self.cod_nodes)]
        tmp_dom = [Node("tmp_dom", i=i) for i, n in enumerate(other.dom_nodes)]
        mapping = {n: n.shift_j(len(self.boxes)) for n in other.box_nodes}
        mapping.update(dict(zip(other.dom_nodes, tmp_dom)))
        positions = {
            n: p.shift(y=other.height + 1) for n, p in self.positions.items()}
        result = self.relabel_nodes(
            dict(zip(self.cod_nodes, tmp_cod)), positions).union(
                other.relabel_nodes(mapping), dom, cod)
        cut = other.height + 0.5
        for i, (u, v) in enumerate(zip(self.cod_nodes, other.dom_nodes)):
            top = result.positions[tmp_cod[i]].x
            bot = result.positions[tmp_dom[i]].x
            if top > bot:
                result.make_space(top - bot, bot, 0, cut)
            elif top < bot:
                result.make_space(bot - top, top, cut, result.height)
            source, = self.graph.predecessors(u)
            target, = other.graph.successors(v)
            result.add_edges([(source, mapping.get(target, target))])
        result.graph.remove_nodes_from(tmp_dom + tmp_cod)
        [result.positions.pop(n) for n in tmp_dom + tmp_cod]
        result = result.relabel_nodes(positions={
            n: p.shift(y=-1)
            for n, p in result.positions.items() if p.y > other.height})
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
        mapping = {n: n.shift_j(len(self.boxes)) for n in other.box_nodes}
        mapping.update({n: n.shift_i(len(self.dom)) for n in other.dom_nodes})
        mapping.update({n: n.shift_i(len(self.cod)) for n in other.cod_nodes})
        if self.height < other.height:
            self = self.stretch(other.height - self.height)
        elif self.height > other.height:
            other = other.stretch(self.height - other.height)
        return self.union(other.relabel_nodes(mapping, positions={
            n: p.shift(x=self.width + 1) for n, p in other.positions.items()}),
            dom=self.dom @ other.dom,
            cod=self.cod @ other.cod)

    def dagger(self) -> Drawing:
        """ The reflection of a drawing along the the horizontal axis. """
        if self.is_box:
            return Drawing.from_box(self.boxes[0].dagger())
        mapping = {n: Node("box", box=n.box[::-1], j=len(self.boxes) - n.j - 1)
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
        dom, cod, boxes = self.cod, self.dom, self.boxes[::-1]
        return Drawing(inside, dom, cod, boxes, self.width, self.height)

    def bubble(self, dom=None, cod=None, name=None, horizontal=False,
               width=None, height=None, draw_as_frame=False) -> Drawing:
        """
        >>> from discopy.symmetric import *
        >>> a, b, c, d = map(Ty, "abcd")
        >>> d = Box('f', a @ b, c @ d).to_drawing().bubble(d @ c @ c, b @ a @ a, "g", draw_as_frame=True)

        # >>> d.draw(path="")
        """
        from discopy.monoidal import Box
        dom = self.dom if dom is None else dom
        cod = self.cod if cod is None else cod
        arg_dom, arg_cod = self.dom, self.cod
        wires_can_go_straight = (
            len(dom), len(cod)) == (len(arg_dom), len(arg_cod))
        draw_as_frame = draw_as_frame or not wires_can_go_straight
        left, right = type(dom)(name or ""), type(dom)("")
        left.inside[0].always_draw_label = True
        top = Box("top", dom, left @ arg_dom @ right).to_drawing()
        bot = Box("bot", left @ arg_cod @ right, cod).to_drawing()
        top.box.draw_as_wires = bot.box.draw_as_wires = True
        # if width is not None:
        #     top.make_space(width - top.width, top.width / 2)
        #     bot.make_space(width - bot.width, bot.width / 2)
        middle = self if height is None else self.stretch(height - self.height)
        result = top >> left @ middle @ right >> bot
        dom_nodes, arg_dom_nodes = ([
                Node(kind, x=x, i=i + off, j=0)
                for i, x in enumerate(xs.inside)]
            for (kind, xs, off) in [
                ("box_dom", dom, 0), ("box_cod", arg_dom, 1)])
        arg_cod_nodes, cod_nodes = ([
                Node(kind, x=x, i=i + off, j=len(result.boxes) - 1)
                for i, x in enumerate(xs.inside)]
            for (kind, xs, off) in [
                ("box_dom", arg_cod, 1), ("box_cod", cod, 0)])
        left_box_cod, right_box_cod = (
            Node("box_cod", j=0, x=x, i=i)
            for i, xs in [(0, left), (len(arg_dom) + 1, right)]
            for x in xs.inside)
        left_box_dom, right_box_dom = (
            Node("box_dom", j=len(result.boxes) - 1, x=x, i=i)
            for i, xs in [(0, left), (len(arg_cod) + 1, right)]
            for x in xs.inside)
        if draw_as_frame:
            result.relabel_nodes(copy=False, positions={
                n: result.positions[n].shift(y=-.25) for n in dom_nodes})
            result.relabel_nodes(copy=False, positions={
                n: result.positions[n].shift(y=.25) for n in cod_nodes})
            result.relabel_nodes(copy=False, positions={
                n: result.positions[n].shift(y=.25)
                for n in arg_dom_nodes + [left_box_cod, right_box_cod]})
            result.relabel_nodes(copy=False, positions={
                n: result.positions[n].shift(y=-.25)
                for n in arg_cod_nodes + [left_box_dom, right_box_dom]})
        if len(dom) == len(arg_dom):
            node = Node("box", j=0, box=top.box)
            result.graph.add_edges_from(zip(top_dom_nodes, top_cod_nodes))
            result.graph.remove_edges_from([(x, node) for x in top_dom_nodes])
            result.graph.remove_edges_from([(node, x) for x in top_cod_nodes])
        if len(cod) == len(arg_cod):
            node = Node("box", j=len(result.boxes) - 1, box=bot.box)
            result.graph.add_edges_from(zip(bot_dom_nodes, bot_cod_nodes))
            result.graph.remove_edges_from([(x, node) for x in bot_dom_nodes])
            result.graph.remove_edges_from([(node, x) for x in bot_cod_nodes])
        return result

    def frame(self, *others: Drawing,
              dom=None, cod=None, name=None, horizontal=False) -> Drawing:
        """
        >>> from discopy.monoidal import *
        >>> x, y = Ty('x'), Ty('y')
        >>> f, g, h = Box('f', x, y ** 3), Box('g', y, y @ y), Box('h', x, y)
        >>> Bubble(f, g, h >> h[::-1], dom=x, cod=y @ y, draw_horizontal=True).draw(aspect='equal')
        >>> Bubble(f, g, h, dom=x, cod=y @ y, draw_horizontal=False).draw(aspect='equal')
        """
        args, empty = (self, ) + others, self.dom[:0]
        width = max([arg.width for arg in args] + [1])
        height = max([arg.height for arg in args] + [1])
        method = getattr(Drawing.id(empty), "tensor" if horizontal else "then")
        return method(*(arg.bubble(
            empty, empty, draw_as_frame=True, height=height, width=width)
            for arg in args)).bubble(dom, cod, draw_as_frame=True)
