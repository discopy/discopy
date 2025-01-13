"""
The category of labeled progressive plane graphs.

This was first defined in :cite:t:`JoyalStreet88`.

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

Axioms
------

* Associativity and unit

>>> from discopy.monoidal import Ty, Box

>>> x, y, z, w = map(Ty, "xyzw")
>>> f = Box('f', x, y).to_drawing()
>>> g = Box('g', y, z).to_drawing()
>>> h = Box('h', z, w).to_drawing()

>>> assert (f >> g) >> h == f >> (g >> h)
>>> assert (f @ g) @ h == f @ (g @ h)

>>> assert f >> Drawing.id(f.cod) == f == Drawing.id(f.dom) >> f
>>> assert f @ Drawing.id() == f == Drawing.id() @ f

* Interchanger

>>> f0, f1 = (Box(f'f{i}', f'x{i}', f'y{i}').to_drawing() for i in (0, 1))
>>> g0, g1 = (Box(f'g{i}', f'y{i}', f'z{i}').to_drawing() for i in (0, 1))

>>> Equation(f0 @ f1 >> g0 @ g1, (f0 >> g0) @ (f1 >> g1)).draw(
...     path="docs/_static/drawing/interchanger-1.png")

.. image:: /_static/drawing/interchanger-1.png
    :align: center

>>> Equation(f @ g.dom >> f.cod @ g, f @ g, f.dom @ g >> f @ g.cod).draw(
...     path="docs/_static/drawing/interchanger-2.png")

.. image:: /_static/drawing/interchanger-2.png
    :align: center
"""


from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

import networkx as nx

from discopy.drawing import backend, Node, Point
from discopy.config import DRAWING_ATTRIBUTES
from discopy.utils import (
    Composable, Whiskerable, assert_isinstance, assert_iscomposable, unbiased)

if TYPE_CHECKING:
    from discopy import monoidal


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

    def nodes_of_kind(self, kind):
        return [node for node in self.nodes if node.kind == kind]

    box_nodes = property(lambda self: self.nodes_of_kind("box"))
    dom_nodes = property(lambda self: self.nodes_of_kind("dom"))
    cod_nodes = property(lambda self: self.nodes_of_kind("cod"))
    box_dom_nodes = property(lambda self: self.nodes_of_kind("box_dom"))
    box_cod_nodes = property(lambda self: self.nodes_of_kind("box_cod"))

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
        """ Call :meth:`add_box_corners` then :func:`backend.draw`. """
        asymmetry = params.pop("asymmetry", 0.125 * any(
            box.is_conjugate or box.is_transpose or (
                box.is_dagger and not box.draw_as_braid)
            for box in self.boxes))
        self.add_box_corners()
        return backend.draw(self, asymmetry=asymmetry, **params)

    def add_box_corners(self):
        """ Recenter boxes w.r.t their wires then draw the corners. """
        for j, box in enumerate(self.boxes):
            box_node = Node("box", j=j, box=box)
            box_x, box_y = self.positions[box_node]
            box_dom_nodes, box_cod_nodes = ([
                    Node(kind, i=i, j=j, x=x) for i, x in enumerate(xs)]
                for kind, xs in [("box_dom", box.dom), ("box_cod", box.cod)])
            xs = [self.positions[n].x for n in box_dom_nodes + box_cod_nodes]
            left, right = min(xs + [box_x]) - 0.25, max(xs + [box_x]) + 0.25
            self.add_nodes({
                Node(f"box-corner-{a}{b}", j=j): Point(x, box_y + y)
                for a, x in enumerate([left, right])
                for b, y in enumerate([-0.25, 0.25])})
            if box.draw_as_wires or box.draw_as_spider:
                if len(box.dom) == 1 or len(box.cod) == 1:
                    continue
            self.positions[box_node] = Point(
                (right + left) / 2, self.positions[box_node].y)

    def union(self, other, dom, cod, width=None, height=None, _check=True):
        """ Take the union of two drawings, assuming nodes are distinct. """
        graph = nx.union(self.inside.graph, other.inside.graph)
        inside = PlaneGraph(graph, self.positions | other.positions)
        boxes = self.boxes + other.boxes
        width = width or max(self.width, other.width)
        height = height or max(self.height, other.height)
        return Drawing(inside, dom, cod, boxes, width, height, _check)

    def add_nodes(self, positions: dict[Node, Point]):
        """ Add nodes to the graph given their positions. """
        if not positions:
            return
        self.graph.add_nodes_from(positions)
        self.positions.update(positions)
        self.width = max(self.width, max(i for (i, _) in positions.values()))
        self.height = max(self.height, max(j for (_, j) in positions.values()))

    def add_edges(self, edges: list[tuple[Node, Node]]):
        """ Add edges from a list. """
        self.graph.add_edges_from(edges)

    def relabel_nodes(
            self, mapping=dict(), positions=dict(), copy=True, _check=False):
        """ Relabel nodes and/or their positions. """
        graph = nx.relabel_nodes(self.graph, mapping, copy)
        if not copy:
            self.positions.update(positions)
            return self
        positions = {mapping.get(node, node): positions.get(node, pos)
                     for node, pos in self.positions.items()}
        inside = PlaneGraph(graph, positions)
        dom, cod, boxes = self.dom, self.cod, self.boxes
        x, y = self.width, self.height
        return Drawing(inside, dom, cod, boxes, x, y, _check=_check)

    def make_space(self, space, x,
                   y_min=None, y_max=None, exclusive=False, copy=False):
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
        result = self.relabel_nodes(copy=copy, positions={
            n: p.shift(x=((p.x > x if exclusive else p.x >= x) * space))
            for n, p in self.positions.items() if y_min <= p.y <= y_max})
        result.width += space
        return result

    def reposition_box_dom(self, j=0):
        """ Recenter dom nodes, used when drawing frames. """
        box_dom = self.boxes[j].dom
        xs = [self.positions[n].x for n in self.nodes
              if n.kind == "box_cod" and n.j == j]
        box_x = self.positions[self.box_nodes[j]].x
        left, right = min(xs + [box_x]), max(xs + [box_x])
        for i, x in enumerate(box_dom):
            target = Node("box_dom", i=i, j=j, x=x)
            source, = self.graph.predecessors(target)
            for n in (source, target):
                x = (right + left - len(box_dom) + 1) / 2 + i
                self.positions[n] = Point(x, self.positions[n].y)

    def reposition_box_cod(self, j=-1):
        """ Recenter cod nodes to recover legacy behaviour for layers. """
        j = j if j > 0 else len(self.boxes) + j
        box = self.boxes[j]
        if box.bubble_closing and len(box.dom[1:-1]) == len(box.cod):
            return  # Otherwise the wires would bend when coming out.
        xs = [self.positions[n].x for n in self.nodes
              if n.kind == "box_dom" and n.j == j]
        box_x = self.positions[self.box_nodes[j]].x
        left, right = min(xs + [box_x]), max(xs + [box_x])
        for i, x in enumerate(box.cod):
            source = Node("box_cod", i=i, j=j, x=x)
            target, = self.graph.successors(source)
            if target.kind != "cod":
                return  # Otherwise we would have to reposition everything.
            for n in (source, target):
                x = (right + left - len(box.cod) + 1) / 2 + i
                self.positions[n] = Point(x, self.positions[n].y)
            if box.draw_as_spider and len(box.cod) == 1:
                box_node = Node("box", box=box, j=j)
                self.positions[box_node] = Point(x, self.positions[box_node].y)

    def align_box_cod(self, j=-1):
        """ Align outputs with inputs when they have equal number of wires. """
        j = j if j > 0 else len(self.boxes) + j
        box = self.boxes[j]
        for i, (x_dom, x_cod) in enumerate(zip(box.dom, box.cod)):
            dom_node = Node("box_dom", i=i, j=j, x=x_dom)
            cod_node = Node("box_cod", i=i, j=j, x=x_cod)
            target, = self.graph.successors(cod_node)
            if target.kind != "cod":
                return  # Otherwise we would have to reposition everything.
            x, _ = self.positions[dom_node]
            _, y = self.positions[cod_node]
            self.positions[cod_node] = Point(x, y)

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
    def is_layer(self):
        """ Whether the drawing is just one box with wires on both sides. """
        return len(self.boxes) == 1

    @property
    def box(self):
        """ Syntactic sugar for self.boxes[0] when self.is_box """
        if not self.is_layer:
            raise ValueError
        return self.boxes[0]

    @property
    def left_is_whiskered(self):
        """ Whether `self = x @ f` for some non-empty type `x`. """
        if len(self.dom) == 0:
            return False
        left_dom = Node("dom", i=0, x=self.dom[0])
        target, = self.graph.successors(left_dom)
        return target.kind == "cod" and target.i == 0

    @property
    def right_is_whiskered(self):
        """ Whether `self = f @ x` for some non-empty type `x`. """
        if len(self.dom) == 0:
            return False
        right_dom = Node("dom", i=len(self.dom) - 1, x=self.dom[-1])
        target, = self.graph.successors(right_dom)
        return target.kind == "cod" and target.i == len(self.cod) - 1

    @staticmethod
    def from_box(box: "monoidal.Box") -> Drawing:
        """
        Draw a diagram with just one box.

        >>> from discopy.monoidal import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box('f', x, y @ z)
        >>> assert f.to_drawing() == Drawing.from_box(f)
        >>> for ps in f.to_drawing().positions.items(): print(*ps)
        Node('box', box=f, j=0) Point(x=1.0, y=0.5)
        Node('dom', i=0, x=x) Point(x=1.0, y=1)
        Node('box_dom', i=0, j=0, x=x) Point(x=1.0, y=0.75)
        Node('box_cod', i=0, j=0, x=y) Point(x=0.5, y=0.25)
        Node('box_cod', i=1, j=0, x=z) Point(x=1.5, y=0.25)
        Node('cod', i=0, x=y) Point(x=0.5, y=0)
        Node('cod', i=1, x=z) Point(x=1.5, y=0)
        >>> f.draw(path="docs/_static/drawing/box.png")

        .. image:: /_static/drawing/box.png
            :align: center
        """
        from discopy.monoidal import Box
        box_dom, box_cod = box.dom.to_drawing(), box.cod.to_drawing()
        old_box, box = box, Box(
            box.name, box_dom, box_cod, is_dagger=box.is_dagger)

        for attr, default in DRAWING_ATTRIBUTES.items():
            setattr(box, attr, getattr(old_box, attr, default(box)))

        if box.draw_as_wires and not box.frame_boundary:
            for i, obj in enumerate(box.cod.inside):
                obj.reposition_label = 0.5 if (
                    box.bubble_closing or box.bubble_opening and i) else 0.25

        if box.bubble_opening:
            width = max(1, len(box.dom), len(box.cod) - 2) + 0.5
        elif box.bubble_closing:
            width = max(1, len(box.dom) - 2, len(box.cod)) + 0.5
        elif len(box.dom) <= 1 and len(box.cod) <= 1:
            width = 1
        else:
            width = max(len(box.dom), len(box.cod))

        height = box.height

        left, right = 0.25, width - 0.25

        inside = PlaneGraph(nx.DiGraph(), dict())
        result = Drawing(
            inside, box.dom, box.cod, (box, ), width, height, _check=False)

        box_node = Node("box", box=box, j=0)
        result.add_nodes({box_node: Point(width / 2, height / 2)})

        dom = [Node("dom", i=i, x=x) for i, x in enumerate(box.dom)]
        cod = [Node("cod", i=i, x=x) for i, x in enumerate(box.cod)]
        box_dom = [Node("box_dom", i=i, j=0, x=x.x) for i, x in enumerate(dom)]
        box_cod = [Node("box_cod", i=i, j=0, x=x.x) for i, x in enumerate(cod)]

        result.add_edges(list(zip(dom, box_dom)))
        result.add_edges([(x, box_node) for x in box_dom])
        result.add_edges([(box_node, x) for x in box_cod])
        result.add_edges(list(zip(box_cod, cod)))

        if box.bubble_opening:
            result.add_nodes({
                cod[0]: Point(left, 0),
                box_cod[0]: Point(left, 0),
                box_cod[-1]: Point(right, 0),
                cod[-1]: Point(right, 0)})
            cod, box_cod = cod[1:-1], box_cod[1:-1]
        elif box.bubble_closing:
            result.add_nodes({
                dom[0]: Point(left, height),
                box_dom[0]: Point(left, height),
                box_dom[-1]: Point(right, height),
                dom[-1]: Point(right, height)})
            dom, box_dom = dom[1:-1], box_dom[1:-1]

        result.add_nodes({
            x: Point(i + (width - len(xs) + 1) / 2, y) for xs, y in [
                (dom, height),
                (box_dom, height if box.draw_as_wires else height - 0.25),
                (box_cod, 0 if box.draw_as_wires else 0.25),
                (cod, 0)]
            for i, x in enumerate(xs)})
        return result

    @staticmethod
    def id(dom: "monoidal.Ty" = None, length=0) -> Drawing:
        """
        Draw the identity diagram.

        >>> from discopy.monoidal import Ty
        >>> Drawing.id(Ty()).draw(path="docs/_static/drawing/empty.png")

        .. image:: /_static/drawing/empty.png
            :align: center

        >>> Drawing.id(Ty('x')).draw(path="docs/_static/drawing/idx.png")

        .. image:: /_static/drawing/idx.png
            :align: center

        >>> Drawing.id(Ty('x', 'y')).draw(path="docs/_static/drawing/idxy.png")

        .. image:: /_static/drawing/idxy.png
            :align: center
        """
        from discopy.monoidal import Ty
        dom = Ty() if dom is None else dom
        inside = PlaneGraph(nx.DiGraph(), dict())
        height, width = 0.5, len(dom) - 0.5 if len(dom) > 1 else 0.5
        result = Drawing(inside, dom, dom, (), width, height, _check=False)
        dom_nodes = [Node("dom", i=i, x=x) for i, x in enumerate(dom)]
        cod_nodes = [Node("cod", i=i, x=x) for i, x in enumerate(dom)]
        result.add_nodes({
            x: Point(i + 0.25, 1) for i, x in enumerate(dom_nodes)})
        result.add_nodes({
            x: Point(i + 0.25, 0) for i, x in enumerate(cod_nodes)})
        result.add_edges(list(zip(dom_nodes, cod_nodes)))
        return result

    @unbiased
    def then(self, other: Drawing, draw_step_by_step=False) -> Drawing:
        """
        Draw one diagram composed with another.

        This is done by calling :meth:`make_space` to align the output wires of
        `self` and the input wires of `other` while keeping them straight.

        Example
        -------
        >>> from discopy.monoidal import Ty, Box, Diagram
        >>> x = Ty('x')
        >>> f = Drawing.from_box(Box('f', x, x @ x @ x))
        >>> g = Drawing.from_box(Box('g', x @ x, x))
        >>> u = Drawing.from_box(Box('u', Ty(), x ** 3))
        >>> v = Drawing.from_box(Box('v', x ** 7, Ty()))

        >>> top, bottom = u >> g @ f, g @ f @ f >> v
        >>> Diagram.to_gif(
        ...     *top.then(bottom, draw_step_by_step=True), loop=True,
        ...     wire_labels=False, draw_box_labels=False,
        ...     path="docs/_static/drawing/composition.gif")
        <IPython.core.display.HTML object>

        .. image:: /_static/drawing/composition.gif
            :align: center
        """
        assert_iscomposable(self, other)
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        dom, cod = self.dom, other.cod

        tmp_cod = [Node("tmp_cod", i=i) for i, n in enumerate(self.cod_nodes)]
        mapping = dict(zip(self.cod_nodes, tmp_cod))
        positions = {
            n: p.shift(y=other.height + 1) for n, p in self.positions.items()}

        tmp_dom = [Node("tmp_dom", i=i) for i, n in enumerate(other.dom_nodes)]
        other_mapping = dict(zip(other.dom_nodes, tmp_dom))
        other_mapping.update({
            n: n.shift_j(len(self.boxes))
            for n in other.nodes if "box" in n.kind})

        result = self.relabel_nodes(mapping, positions, _check=False).union(
                other.relabel_nodes(other_mapping), dom, cod, _check=False)
        result.height = self.height + other.height + 1

        cut, top_width, bot_width = other.height + 0.5, self.width, other.width
        if draw_step_by_step:
            steps = [result.relabel_nodes(copy=True)]
        for i, (u, v) in enumerate(zip(self.cod_nodes, other.dom_nodes)):
            top = result.positions[tmp_cod[i]].x
            bot = result.positions[tmp_dom[i]].x
            if top > bot:
                bot_width += top - bot
                result.make_space(top - bot, (i > 0) * bot, 0, cut)
            elif top < bot:
                top_width += bot - top
                result.make_space(bot - top, (i > 0) * top, cut, result.height)
            source, = self.graph.predecessors(u)
            target, = other.graph.successors(v)
            result.add_edges([(source, other_mapping.get(target, target))])
            if draw_step_by_step:
                steps.append(result.relabel_nodes(copy=True))

        result.graph.remove_nodes_from(tmp_dom + tmp_cod)
        [result.positions.pop(n) for n in tmp_dom + tmp_cod]
        result.relabel_nodes(copy=False, positions={
            n: p.shift(y=-1)
            for n, p in result.positions.items() if p.y > other.height})
        result.height = self.height + other.height
        result.width = max(top_width, bot_width)
        for j, box in enumerate(other.boxes):  # Recover legacy behaviour.
            if len(box.dom) == len(box.cod) and not box.frame_boundary:
                result.align_box_cod(len(self.boxes) + j)
            else:
                result.reposition_box_cod(len(self.boxes) + j)
        if draw_step_by_step:
            for step in steps:
                step.width = result.width
        return steps if draw_step_by_step else result

    def stretch(self, y, copy=True):
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
        result = self.relabel_nodes(copy=copy, positions={n: p.shift(y=(
                y if n.kind == "dom" else 0 if n.kind == "cod" else y / 2))
            for n, p in self.positions.items()})
        result.height += y
        return result

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
            n: n.shift_j(len(self.boxes))
            for n in other.nodes if "box" in n.kind}
        mapping.update({n: n.shift_i(len(self.dom)) for n in other.dom_nodes})
        mapping.update({n: n.shift_i(len(self.cod)) for n in other.cod_nodes})
        if self.height < other.height:
            self = self.stretch(other.height - self.height)
        elif self.height > other.height:
            other = other.stretch(self.height - other.height)
        x_shift = self.width + (
            0.25 if self.right_is_whiskered or other.left_is_whiskered else 0)
        result = self.union(other.relabel_nodes(mapping, positions={
            n: p.shift(x=x_shift) for n, p in other.positions.items()}),
            dom=self.dom @ other.dom, cod=self.cod @ other.cod, _check=False)
        result.width = x_shift + other.width
        return result

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
    def bubble_opening(dom, arg_dom, left, right, frame_boundary=False):
        """
        Construct the opening of a bubble, i.e. a box drawn as wires.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.bubble_opening(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/bubble-opening.png")

        .. image:: /_static/drawing/bubble-opening.png
            :align: center
        """
        from discopy.monoidal import Box
        return Box(
            "top", dom, left @ arg_dom @ right,
            bubble_opening=True, frame_boundary=frame_boundary,
            height=(0.5 if frame_boundary else 1)).to_drawing()

    @staticmethod
    def bubble_closing(arg_cod, cod, left, right, frame_boundary=False):
        """
        Construct the closing of a bubble, i.e. a box drawn as wires.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.bubble_closing(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/bubble-closing.png")

        .. image:: /_static/drawing/bubble-closing.png
            :align: center
        """
        from discopy.monoidal import Box
        return Box(
            "bot", left @ arg_cod @ right, cod,
            bubble_closing=True, frame_boundary=frame_boundary,
            height=(0.5 if frame_boundary else 1)).to_drawing()

    @staticmethod
    def frame_opening(dom, arg_dom, left, right):
        """
        Construct the opening of a frame as the opening of a bubble squashed to
        zero height so that it looks like the upper half of a rectangle.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.frame_opening(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/frame-opening.png")

        .. image:: /_static/drawing/frame-opening.png
            :align: center
        """
        result = Drawing.bubble_opening(
            dom, arg_dom, left, right, frame_boundary=True)
        box_dom_nodes = result.box_dom_nodes
        box_cod_nodes = result.box_cod_nodes
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-0.25) for n in box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=0.25) for n in box_cod_nodes})
        result.graph.remove_edges_from([
            (u, v) for u in box_dom_nodes for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in box_cod_nodes[1:-1]])
        return result

    @staticmethod
    def frame_closing(arg_cod, cod, left, right):
        """
        Construct the closing of a frame as the closing of a bubble squashed to
        zero height so that it looks like the lower half of a rectangle.

        >>> from discopy.monoidal import Ty
        >>> x, y, z = map(Ty, "xyz")
        >>> Drawing.frame_closing(x, y, z, Ty("")).draw(
        ...     path="docs/_static/drawing/frame-closing.png")

        .. image:: /_static/drawing/frame-closing.png
            :align: center
        """
        result = Drawing.bubble_closing(
            arg_cod, cod, left, right, frame_boundary=True)
        box_dom_nodes = result.box_dom_nodes
        box_cod_nodes = result.box_cod_nodes
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=-0.25) for n in box_dom_nodes})
        result.relabel_nodes(copy=False, positions={
            n: result.positions[n].shift(y=0.25) for n in box_cod_nodes})
        result.graph.remove_edges_from([
            (u, v) for u in box_dom_nodes[1:-1] for v in result.box_nodes] + [
            (u, v) for u in result.box_nodes for v in box_cod_nodes])
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
        result.make_space(-0.25, 0.25, exclusive=True)
        result.make_space(-0.25, result.width - 0.25)
        if width is not None and result.width != width:
            if result.width > width:
                raise ValueError
            space = (width - result.width) / 2
            result.make_space(space, 0.25, exclusive=True)
            result.make_space(space, result.width - 0.25)
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
        from discopy.monoidal import Ty
        args = (self, ) + others
        method = "then" if draw_vertically else "tensor"
        params = dict(
                width=max([arg.width for arg in args] + [0]) + 1
            ) if draw_vertically else dict(
                height=max([arg.height for arg in args] + [0]))
        result = getattr(Drawing.id(), method)(*(arg.bubble(
            Ty(), Ty(), draw_as_square=True, **params)
            for arg in args)).bubble(dom, cod, name, draw_as_square=True)
        result.reposition_box_dom()
        result.reposition_box_cod()
        return result

    def zero(dom, cod):
        from discopy.monoidal import Box
        result = Box("zero", dom, cod).to_drawing()
        result.zero_drawing = True
        return result

    def add(self, other: Drawing, symbol="+", space=1):
        """ Concatenate two drawings with a symbol in between. """
        from discopy.monoidal import Ty, Box
        if getattr(self, "zero_drawing", False):
            return other
        if getattr(other, "zero_drawing", False):
            return self
        scalar = Box(symbol, Ty(), Ty(), draw_as_spider=True, color="white")
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
    ...          aspect='equal', wire_labels=False,
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
