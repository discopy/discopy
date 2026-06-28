# -*- coding: utf-8 -*-
"""
DisCopy's drawing backends: Matplotlib and TikZ.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    draw
    Backend
    TikZ
    Matplotlib
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch
from matplotlib.path import Path

from discopy.drawing import Node, Point

from discopy.config import (  # noqa: F401
    DRAWING_ATTRIBUTES as ATTRIBUTES,
    DRAWING_DEFAULT as DEFAULT, COLORS, SHAPES)

if TYPE_CHECKING:
    from discopy.drawing import PlaneGraph


def draw(graph: PlaneGraph, **params):
    """ Load a :class:`Backend` and draw a :class:`PlaneGraph` on it. """
    aspect = params.get('aspect', 'auto' if 'figsize' in params else 'equal')
    figsize = params.get('figsize', None if aspect == 'auto' else (
        graph.width or 1, graph.height or 1))
    backend = (
        TikZ(use_tikzstyles=params.get('use_tikzstyles', None))
        if params.get('to_tikz', False)
        else Matplotlib(figsize=figsize,
                        linewidth=params.get('linewidth', 1)))

    max_v = max(graph.height, graph.width, 0.01)
    params['nodesize'] = round(params.get('nodesize', 1.) / sqrt(max_v), 3)

    backend.draw_boundary(graph, **params)
    backend.draw_ribbons(graph, **params)
    backend.draw_wires(graph, **params)
    backend.draw_boxes(graph, **params)
    backend.draw_spiders(graph, **params)

    return backend.output(
        path=params.get('path', None),
        baseline=graph.height / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True), aspect=aspect,
        margins=params.get('margins', DEFAULT['margins']))


def _bezier_subcurve(points, t0, t1):
    """ Restrict a cubic Bezier (4 control points) to the range [t0, t1]. """
    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def split(p, t):  # The two halves of a cubic Bezier split at ``t``.
        a, b, c = lerp(p[0], p[1], t), lerp(p[1], p[2], t), lerp(p[2], p[3], t)
        d, e = lerp(a, b, t), lerp(b, c, t)
        f = lerp(d, e, t)
        return [p[0], a, d, f], [f, e, c, p[3]]

    right = split(points, t0)[1]
    return split(right, (t1 - t0) / (1 - t0))[0]


class Backend(ABC):
    """ Abstract drawing backend. """
    def __init__(self, linewidth=1):
        self.max_width = 0

    def draw_text(self, text, i, j, **params):
        """ Draws a piece of text at a given position. """
        self.max_width = max(self.max_width, i)

    def draw_node(self, i, j, **params):
        """ Draws a node for a given position, color and shape. """
        self.max_width = max(self.max_width, i)

    def draw_polygon(self, *points, facecolor=None, edgecolor=None):
        """ Draws a polygon given a list of points. """
        self.max_width = max(self.max_width, max(i for i, _ in points))

    def draw_wire(self, source, target,
                  bend_out=False, bend_in=False, style=None):
        """ Draws a wire from source to target, possibly with a Bezier. """
        self.max_width = max(self.max_width, source[0], target[0])

    def draw_bezier(self, points):
        """ Draws a cubic Bezier curve from a list of four control points. """
        self.max_width = max(self.max_width, max(x for x, _ in points))

    def draw_filled_shape(self, start, steps, color):
        """
        Fills the closed region whose boundary starts at ``start`` and follows
        ``steps``, a list of either ``("line", end)`` for a straight segment or
        ``("curve", control1, control2, end)`` for a cubic Bezier. The region
        is filled with ``color`` and drawn without an outline, e.g. behind the
        wires to colour the inside of a ribbon.
        """
        points = [start] + [step[-1] for step in steps]
        self.max_width = max([self.max_width] + [x for x, _ in points])

    @staticmethod
    def _ribbon(typ):
        # The Ribbon a wire belongs to, shared by its two rails, or None.
        inside = getattr(typ, "inside", None)
        return getattr(inside[0], "ribbon", None) if inside else None

    @staticmethod
    def _ribbon_color(typ):
        # The colour filling the ribbon a wire belongs to, or None.
        ribbon = Backend._ribbon(typ)
        return None if ribbon is None else ribbon.color

    def draw_ribbons(self, graph, **params):
        """
        Fills the inside of the straight parts of the ribbons in a dual rail
        drawing, i.e. the band between the two rails of each ribbon. The bends,
        crossings and folds are coloured by the boxes drawing them, see e.g.
        :meth:`draw_dual_rail_braid`.
        """
        positions = graph.positions
        wire_kinds = ("dom", "cod", "box_dom", "box_cod")
        wires = {(n.kind, getattr(n, "j", None), n.i): n
                 for n in graph.nodes if n.kind in wire_kinds}
        succ = {s: t for s, t in graph.edges() if s.kind in wire_kinds}
        for node in graph.nodes:
            if node.kind not in ("dom", "box_cod"):
                continue  # The two endpoints below a wire flow downwards.
            ribbon = self._ribbon(getattr(node, "x", None))
            if ribbon is None or ribbon.color is None:
                continue  # Not a coloured rail, i.e. nothing to fill.
            # The other rail of the ribbon, sharing it on either side.
            partner = next((other for other in (
                wires.get((node.kind, getattr(node, "j", None), node.i + di))
                for di in (-1, 1)) if other is not None
                and self._ribbon(getattr(other, "x", None)) is ribbon), None)
            if partner is None or positions[node][0] > positions[partner][0]:
                continue  # Fill the band once, from its left rail.
            target, partner_target = succ.get(node), succ.get(partner)
            if target is None or partner_target is None:
                continue
            if target.kind == "box" or partner_target.kind == "box":
                continue  # The wire runs into a box, drawn on its own.
            self.draw_filled_shape(positions[node], [
                ("line", positions[target]),
                ("line", positions[partner_target]),
                ("line", positions[partner])], ribbon.color)

    def _strand(self, source, target, middle):
        # The four control points of a braid strand, see draw_braid_strand.
        return [source, (source[0], middle), (target[0], middle), target]

    def _fill_strand_band(self, first, second, middle, color, gap=0):
        # Fills the band between two parallel braid strands of a ribbon. A
        # non-zero gap breaks the band around the crossing, matching the broken
        # strands, so the ribbon going under is shadowed by the one going over.
        a, b = self._strand(*first, middle), self._strand(*second, middle)
        spans = [(0, 1)] if not gap else [(0, 0.5 - gap), (0.5 + gap, 1)]
        for t0, t1 in spans:
            a_sub, b_sub = (
                _bezier_subcurve(a, t0, t1), _bezier_subcurve(b, t0, t1))
            self.draw_filled_shape(a_sub[0], [
                ("curve", a_sub[1], a_sub[2], a_sub[3]), ("line", b_sub[3]),
                ("curve", b_sub[2], b_sub[1], b_sub[0])], color)

    def _half_circle_beziers(self, left, right, end, sign):
        # The two Bezier control groups of the arc of a half circle, see
        # _half_circle, from (left, end) to (right, end) bulging by ``sign``.
        middle, radius = (left + right) / 2, (right - left) / 2
        k = radius * 4 * (sqrt(2) - 1) / 3
        apex = end + sign * radius
        return ([(left, end), (left, end + sign * k),
                 (middle - k, apex), (middle, apex)],
                [(middle, apex), (middle + k, apex),
                 (right, end + sign * k), (right, end)])

    def draw_braid_strand(self, source, target, middle, gap=0):
        """
        Draws a single strand of a braid crossing the horizontal line at
        height ``middle``. The strand is vertical at both ends and diagonal in
        between, so that two strands cross at a right angle rather than meeting
        flat. If ``gap`` is non-zero the strand is broken around the crossing,
        i.e. it goes under the other strand.
        """
        control = [source, (source[0], middle), (target[0], middle), target]
        if not gap:
            return self.draw_bezier(control)
        self.draw_bezier(_bezier_subcurve(control, 0, 0.5 - gap))
        self.draw_bezier(_bezier_subcurve(control, 0.5 + gap, 1))

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        """ Draws a list of boxes depicted as spiders. """
        spider_widths = [
            p.x for n, p in graph.positions.items()
            if n.kind == 'box' and n.box.draw_as_spider]
        if spider_widths:
            self.max_width = max(self.max_width, max(spider_widths))

    @abstractmethod
    def output(self, path=None, show=True, **params):
        """ Output the drawing. """

    def draw_boundary(self, graph, boundary_color="white", **params):
        x, y = graph.width, graph.height
        self.draw_polygon(
            (0, 0), (x, 0), (x, y), (0, y), edgecolor=boundary_color)

    def draw_wire_label(self, x, i, j, **params):
        draw_label_anyway = params.get('draw_box_labels', True) and getattr(
            x, "always_draw_label", False)
        if not params.get('wire_labels', True) and not draw_label_anyway:
            return
        if hasattr(x.inside[0], "reposition_label"):
            j += 0.25  # The label of e.g. cups, caps and swaps.
        label = str(x.inside[0])
        pad_i, pad_j = params.get('textpad', DEFAULT['textpad'])
        i += pad_i
        j -= pad_j
        fontsize = params.get('fontsize_types', params.get('fontsize', None))
        self.draw_text(label, i, j, verticalalignment='top',
                       family='monospace', fontsize=fontsize)

    @staticmethod
    def _is_crossing(box):
        # A braid or a swap, i.e. a box whose two wires cross over each other.
        if getattr(box, "draw_as_braid", False):
            return True
        return box.draw_as_wires and len(box.dom) == 2 == len(box.cod)\
            and not box.bubble_opening and not box.bubble_closing

    def draw_wires(self, graph, **params):
        # Braids and swaps cross their two wires diagonally, see draw_braid.
        for node in graph.nodes:
            if node.kind == "box" and self._is_crossing(node.box):
                self.draw_braid(graph.positions, node)
            elif node.kind == "box" and self._is_cup_or_cap(node.box):
                self.draw_cup_or_cap(graph.positions, node)
        for source, target in graph.edges():
            def inside_a_box(node):
                return node.kind == "box"\
                    and not node.box.draw_as_wires\
                    and not node.box.draw_as_spider
            if inside_a_box(source) or inside_a_box(target):
                continue  # no need to draw wires inside a box
            source_position = graph.positions[source]
            target_position = graph.positions[target]
            if source.kind in ["dom", "box_cod"]:
                self.draw_wire_label(source.x, *source_position, **params)
            if source_position == target_position:
                continue
            if any(n.kind == "box" and (self._is_crossing(n.box)
                   or self._is_cup_or_cap(n.box)) for n in (source, target)):
                continue  # cups, caps and crossings are drawn on their own
            bend_out, bend_in = source.kind == "box", target.kind == "box"
            self.draw_wire(
                source_position, target_position, bend_out, bend_in)

    @staticmethod
    def _is_cup_or_cap(box):
        # A cup (two inputs, no output) or a cap (no input, two outputs).
        if not box.draw_as_wires\
                or box.bubble_opening or box.bubble_closing:
            return False
        return sorted((len(box.dom), len(box.cod))) == [0, 2]

    @staticmethod
    def _cup_ends(positions, node):
        # The left end, right end and their shared height for a cup or cap.
        box, j = node.box, node.j
        kind, wires = ("box_dom", box.dom) if box.dom else ("box_cod", box.cod)
        xs = [positions[Node(kind, i=i, j=j, x=wires[i])] for i in range(2)]
        return min(p[0] for p in xs), max(p[0] for p in xs), xs[0][1]

    def draw_cup_or_cap(self, positions, node):
        """
        Draws a cup or a cap as a half circle with vertical sides. Nested ones
        (e.g. the rails of a ribbon) share the centre of the cup or cap they
        sit in, so that they stay concentric, i.e. at a constant distance.
        """
        left, right, end = self._cup_ends(positions, node)
        radius = (right - left) / 2
        down = bool(node.box.dom)  # A cup opens upwards, its arc points down.
        # Share the centre of the tightest cup or cap we are nested in.
        centre, span = end, float("inf")
        for other in positions:
            if other.kind != "box" or other is node\
                    or not self._is_cup_or_cap(other.box)\
                    or bool(other.box.dom) != down:
                continue
            o_left, o_right, o_end = self._cup_ends(positions, other)
            if o_left < left and right < o_right and o_right - o_left < span:
                centre, span = o_end, o_right - o_left
        height = 2 * abs(end - positions[node][1])
        if radius > height:  # Too wide to fit a half circle: fall back to a U.
            control = (4 * positions[node][1] - end) / 3
            return self.draw_bezier(
                [(left, end), (left, control), (right, control), (right, end)])
        self._half_circle(left, right, end, centre, -1 if down else 1)

    def _half_circle(self, left, right, end, centre, sign):
        # A half circle from (left, end) to (right, end) with vertical sides
        # down to ``centre``, drawn as two quarters with the Bezier constant.
        middle, radius = (left + right) / 2, (right - left) / 2
        k = radius * 4 * (sqrt(2) - 1) / 3
        if end != centre:
            self.draw_wire((left, end), (left, centre))
            self.draw_wire((right, centre), (right, end))
        self.draw_bezier([
            (left, centre), (left, centre + sign * k),
            (middle - k, centre + sign * radius),
            (middle, centre + sign * radius)])
        self.draw_bezier([
            (middle, centre + sign * radius),
            (middle + k, centre + sign * radius),
            (right, centre + sign * k), (right, centre)])

    def draw_dual_rail_cup(self, positions, node, **params):
        """
        Draws a :class:`discopy.ribbon.DualRailCup` (or cap) as a single
        constant-width fold, i.e. two concentric half circles joining the outer
        and inner rails of two ribbons.
        """
        box, j = node.box, node.j
        kind, wires = ("box_dom", box.dom) if box.dom else ("box_cod", box.cod)
        xs = [positions[Node(kind, i=i, j=j, x=wires[i])] for i in range(4)]
        end, sign = xs[0][1], -1 if box.dom else 1
        color = self._ribbon_color(wires[:1])
        if color is not None:  # Fill the fold between outer and inner arcs.
            outer = self._half_circle_beziers(xs[0][0], xs[3][0], end, sign)
            inner = self._half_circle_beziers(xs[1][0], xs[2][0], end, sign)
            self.draw_filled_shape(outer[0][0], [
                ("curve", *outer[0][1:]), ("curve", *outer[1][1:]),
                ("line", inner[1][3]),
                ("curve", inner[1][2], inner[1][1], inner[1][0]),
                ("curve", inner[0][2], inner[0][1], inner[0][0])], color)
        for a, b in [(0, 3), (1, 2)]:  # The outer and the inner fold.
            self._half_circle(xs[a][0], xs[b][0], end, end, sign)

    def draw_braid(self, positions, node):
        """
        Draws a braid or a swap as its two wires crossing diagonally, so that
        they meet at a right angle. A braid (over/under) breaks the wire that
        goes under; a symmetric swap simply crosses both wires.
        """
        box, j = node.box, node.j
        dom = [positions[Node("box_dom", i=i, j=j, x=box.dom[i])]
               for i in range(2)]
        cod = [positions[Node("box_cod", i=i, j=j, x=box.cod[i])]
               for i in range(2)]
        _, middle = positions[node]
        left, right = (dom[0], cod[1]), (dom[1], cod[0])
        if not getattr(box, "draw_as_braid", False):
            self.draw_braid_strand(*left, middle)
            self.draw_braid_strand(*right, middle)
            return
        # Keep the shadow roughly the same height, e.g. for a double braid,
        # by widening the (relative) gap when the braid is short.
        gap = min(0.3, 0.1 / (dom[0][1] - cod[0][1]))
        # The left wire goes under the right one unless the box is dagger.
        over, under = (left, right) if box.is_dagger else (right, left)
        self.draw_braid_strand(*under, middle, gap=gap)
        self.draw_braid_strand(*over, middle)

    def draw_boxes(self, graph, **params):
        drawing_methods = [
            ("draw_as_brakets", "draw_brakets"),
            ("draw_as_controlled", "draw_controlled_gate"),
            ("draw_as_discards", "draw_discard"),
            ("draw_as_measures", "draw_measure"),
            ("draw_as_dual_rail_braid", "draw_dual_rail_braid"),
            ("draw_as_dual_rail_twist", "draw_dual_rail_twist"),
            ("draw_as_dual_rail_cup", "draw_dual_rail_cup"),
            (None, "draw_box")]
        box_nodes = [node for node in graph.nodes if node.kind == "box"]
        for node in box_nodes:
            if node.box.draw_as_spider or node.box.draw_as_wires:
                continue
            for attribute, method in drawing_methods:
                if attribute is None or getattr(node.box, attribute, False):
                    getattr(self, method)(graph.positions, node, **params)
                    break

    def draw_box(self, positions, node, **params):
        """ Draws a box node on a given backend. """
        box, j = node.box, node.j
        asymmetry = params.get('asymmetry', 0)
        points = [positions[Node(f"box-corner-{c}", j=j)]
                  for c in ["00", "01", "11", "10"]]
        i = (0 if box.is_conjugate else
             1 if box.is_transpose else
             2 if box.is_dagger else 3)
        if box.is_conjugate or box.is_transpose:
            asymmetry *= -1
        points[i] = points[i].shift(x=asymmetry)
        self.draw_polygon(*points, facecolor=box.color)
        if params.get('draw_box_labels', True):
            self.draw_text(box.drawing_name, *positions[node],
                           ha='center', va='center', family='monospace',
                           fontsize=params.get('fontsize', None))

    def draw_discard(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.circuit.Discard` box. """
        box, j = node.box, node.j
        for i in range(len(box.dom)):
            x = box.dom[i]
            wire = Node("box_dom", x=x, j=j, i=i)
            middle = positions[wire]
            left, right = middle[0] - .25, middle[0] + .25
            height = positions[node][1] + .25
            for j in range(3):
                source = (left + .1 * j, height - .1 * j)
                target = (right - .1 * j, height - .1 * j)
                self.draw_wire(source, target)

    def draw_measure(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.circuit.Measure` box. """
        self.draw_box(positions, node, **dict(params, draw_box_labels=False))
        i, j = positions[node]
        self.draw_wire((i - .15, j - .1), (i, j + .1), bend_in=True)
        self.draw_wire((i, j + .1), (i + .15, j - .1), bend_out=True)
        self.draw_wire((i, j - .1), (i + .05, j + .15), style='->')

    def draw_dual_rail_braid(self, positions, node, **params):
        """
        Draws a :class:`discopy.balanced.DualRailBraid`, i.e. the two ribbons
        ``(0, 1)`` and ``(2, 3)`` crossing as a whole rather than wire by wire.
        """
        box, j = node.box, node.j
        dom = [positions[Node("box_dom", i=i, j=j, x=box.dom[i])]
               for i in range(len(box.dom))]
        cod = [positions[Node("box_cod", i=i, j=j, x=box.cod[i])]
               for i in range(len(box.cod))]
        _, y_middle = positions[node]
        # The left ribbon goes to the right and vice-versa. As with a braid,
        # the left ribbon goes under the right one unless the box is dagger.
        left = ([(dom[0], cod[2]), (dom[1], cod[3])],
                self._ribbon_color(box.dom[:1]))
        right = ([(dom[2], cod[0]), (dom[3], cod[1])],
                 self._ribbon_color(box.dom[2:3]))
        over, under = (left, right) if box.is_dagger else (right, left)
        # Fill (and stroke) under first, then over: the band going under is
        # broken around the crossing so the one going over is clearly on top.
        for (ribbon, color), gap in [(under, 0.2), (over, 0)]:
            if color is not None:  # Fill the band between the two strands.
                self._fill_strand_band(
                    ribbon[0], ribbon[1], y_middle, color, gap)
        for (ribbon, _), gap in [(under, 0.2), (over, 0)]:
            for source, target in ribbon:
                self.draw_braid_strand(source, target, y_middle, gap)

    def draw_dual_rail_twist(self, positions, node, **params):
        """
        Draws a :class:`discopy.balanced.DualRailTwist`, i.e. the two rails of
        a ribbon crossing each other twice in quick succession.
        """
        box, j = node.box, node.j
        dom = [positions[Node("box_dom", i=i, j=j, x=box.dom[i])]
               for i in range(2)]
        cod = [positions[Node("box_cod", i=i, j=j, x=box.cod[i])]
               for i in range(2)]
        _, middle = positions[node]
        # The rails swap at the middle then swap back, i.e. they twist.
        swap = [(dom[1][0], middle), (dom[0][0], middle)]
        upper, lower = (dom[0][1] + middle) / 2, (middle + cod[0][1]) / 2
        color = self._ribbon_color(box.dom[:1])
        if color is not None:  # Fill the band between the two twisting rails.
            rail0 = self._strand(dom[0], swap[0], upper)\
                + self._strand(swap[0], cod[0], lower)[1:]
            rail1 = self._strand(dom[1], swap[1], upper)\
                + self._strand(swap[1], cod[1], lower)[1:]
            self.draw_filled_shape(rail0[0], [
                ("curve", rail0[1], rail0[2], rail0[3]),
                ("curve", rail0[4], rail0[5], rail0[6]),
                ("line", rail1[6]),
                ("curve", rail1[5], rail1[4], rail1[3]),
                ("curve", rail1[2], rail1[1], rail1[0])], color)
        crossings = [
            [(dom[0], swap[0], upper), (dom[1], swap[1], upper)],
            [(swap[0], cod[0], lower), (swap[1], cod[1], lower)]]
        for k, (first_rail, second_rail) in enumerate(crossings):
            # A twist is a braid followed by another of the same handedness
            # (not by its inverse), so the same diagonal stays on top: the
            # rails swap which one goes under between the two crossings.
            first_under = (k == 0) != box.is_dagger
            under, over = (first_rail, second_rail) if first_under\
                else (second_rail, first_rail)
            self.draw_braid_strand(under[0], under[1], under[2], gap=0.15)
            self.draw_braid_strand(over[0], over[1], over[2])

    def draw_brakets(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.gates.Ket` box. """
        box, j = node.box, node.j
        is_bra = len(box.dom) > 0
        for i, bit in enumerate(box._digits):
            kind = "box_dom" if is_bra else "box_cod"
            x = box.dom[i] if is_bra else box.cod[i]
            wire = Node(kind, x=x, j=j, i=i)
            middle = positions[wire]
            left = middle[0] - .25, middle[1]
            right = middle[0] + .25, middle[1]
            top = middle[0], middle[1] + .5
            bottom = middle[0], middle[1] - .5
            self.draw_polygon(
                left, right, bottom if is_bra else top, facecolor=box.color)
            self.draw_text(
                bit, middle[0], middle[1] + (-.25 if is_bra else .2),
                ha='center', va='center',
                fontsize=params.get('fontsize', None))

    def draw_controlled_gate(self, positions, node, **params):
        """ Draws a :class:`discopy.quantum.gates.Controlled` gate. """
        box, j = node.box, node.j
        distance = box.distance
        c_size = len(box.controlled.dom)

        index = (0, distance) if distance > 0 else (c_size - distance - 1, 0)
        dom = Node("box_dom", x=box.dom[0], i=index[0], j=j)
        cod = Node("box_cod", x=box.cod[0], i=index[0], j=j)
        middle = positions[dom][0], (positions[dom][1] + positions[cod][1]) / 2
        controlled_box = box.controlled.to_drawing().box
        controlled = Node("box", box=controlled_box, j=j)
        # TODO select x properly for classical gates
        c_dom = Node("box_dom", x=box.dom[0], i=index[1], j=j)
        c_cod = Node("box_cod", x=box.cod[0], i=index[1], j=j)
        c_middle = Point(
            positions[c_dom][0],
            (positions[c_dom][1] + positions[c_cod][1]) / 2)
        target = Point(
            positions[c_dom][0] + (c_size - 1) / 2,
            (positions[c_dom][1] + positions[c_cod][1]) / 2)
        target_boundary = target
        if controlled_box.name == "X":  # CX gets drawn as a circled plus sign.
            self.draw_wire(positions[c_dom], positions[c_cod])
            eps = 1e-10
            perturbed_target = target[0], target[1] + eps
            self.draw_node(
                *perturbed_target,
                shape="circle", color="white", edgecolor="black",
                nodesize=2 * params.get("nodesize", 1))
            self.draw_node(
                *target, shape="plus",
                nodesize=2 * params.get("nodesize", 1))
        else:
            fake_positions = {controlled: target} | {
                Node(f"box-corner-{a}{b}", j=j): target.shift(x=x, y=y)
                for a, x in enumerate([-0.25, 0.25])
                for b, y in enumerate([-0.25, 0.25])}

            for i in range(c_size):
                dom_node = Node("box_dom", x=box.dom[i], i=i, j=j)
                x, y = positions[c_dom][0] + i, positions[c_dom][1]
                fake_positions[dom_node] = x, y

                cod_node = Node("box_cod", x=box.cod[i], i=i, j=j)
                x, y = positions[c_cod][0] + i, positions[c_cod][1]
                fake_positions[cod_node] = x, y

            shift_boundary = True
            if hasattr(box.controlled, "draw_as_controlled"):
                self.draw_controlled_gate(fake_positions, controlled, **params)

                next_box = box.controlled
                while hasattr(next_box, "controlled"):
                    if controlled_box.distance * next_box.distance < 0:
                        shift_boundary = False
                        break
                    next_box = next_box.controlled
                if next_box.name == "X":
                    shift_boundary = False
            else:
                self.draw_box(fake_positions, controlled, **params)

            if shift_boundary:
                if box.distance > 0:
                    target_boundary = c_middle[0] - .25, c_middle[1]
                else:
                    target_boundary = (
                        c_middle[0] + c_size - 1 + .25, c_middle[1])
            else:
                if box.distance > 0:
                    target_boundary = c_middle[0], c_middle[1]
                else:
                    target_boundary = c_middle[0] + c_size - 1, c_middle[1]
        self.draw_wire(positions[dom], positions[cod])

        # draw all the other vertical wires
        extra_offset = 1 if distance > 0 else len(box.controlled.dom)
        for i in range(extra_offset, extra_offset + abs(distance) - 1):
            node1 = Node("box_dom", x=box.dom[i], i=i, j=j)
            node2 = Node("box_cod", x=box.cod[i], i=i, j=j)
            self.draw_wire(positions[node1], positions[node2])

        # TODO change bend_in and bend_out for tikz backend
        self.draw_wire(middle, target_boundary, bend_in=True, bend_out=True)

        self.draw_node(
            *middle, color="black", shape="circle",
            nodesize=params.get("nodesize", 1))


class TikZ(Backend):
    """ Tikz drawing backend. """
    def __init__(self, use_tikzstyles=None):
        self.use_tikzstyles = DEFAULT["use_tikzstyles"]\
            if use_tikzstyles is None else use_tikzstyles
        self.node_styles, self.edge_styles = [], []
        self.nodes, self.nodelayer, self.edgelayer = {}, [], []
        super().__init__()

    @staticmethod
    def format_color(color):
        """ Formats a color. """
        hexcode = COLORS[color]
        rgb = [
            int(hex, 16) for hex in [hexcode[1:3], hexcode[3:5], hexcode[5:]]]
        return f"{{rgb,255: red,{rgb[0]}; green,{rgb[1]}; blue,{rgb[2]}}}"

    def add_node(self, i, j, text=None, options=None):
        """ Add a node to the tikz picture, return its unique id. """
        node = len(self.nodes) + 1
        text = "" if text is None else text
        self.nodelayer.append(
            f"\\node [{options or ''}] ({node}) at ({i}, {j}) {{{text}}};\n")
        self.nodes.update({(i, j): node})
        return node

    def draw_node(self, i, j, text=None, **params):
        options = []
        if 'shape' in params:
            options.append(params['shape'])
        if 'color' in params:
            options.append(params['color'])
        self.add_node(i, j, text, options=", ".join(options))
        super().draw_node(i, j, **params)

    def draw_text(self, text, i, j, **params):
        options = "style=none, fill=white"
        if params.get('horizontalalignment', 'center') == 'left':
            options += ", anchor=west"
        if params.get("verticalalignment", "center") == "top":  # wire labels
            options += ", right"
        if 'fontsize' in params and params['fontsize'] is not None:
            options += f", scale={params['fontsize']}"
        self.add_node(i, j, text, options)
        super().draw_text(text, i, j, **params)

    def draw_polygon(
            self, *points,
            facecolor=DEFAULT["facecolor"], edgecolor=DEFAULT["edgecolor"]):
        nodes = []
        for point in points:
            nodes.append(self.add_node(*point))
        nodes.append(nodes[0])
        if self.use_tikzstyles:
            style_name = "box" if facecolor == DEFAULT["facecolor"]\
                else f"{facecolor}_box"
            style = f"\\tikzstyle{{{style_name}}}=" \
                    f"[-, fill={self.format_color(facecolor)}]\n"
            if style not in self.edge_styles:
                self.edge_styles.append(style)
            options = f"style={style_name}"
        else:
            options = f"-, fill={{{facecolor}}}"
        str_connections = " to ".join(f"({node}.center)" for node in nodes)
        self.edgelayer.append(f"\\draw [{options}] {str_connections};\n")
        super().draw_polygon(*points)

    def draw_wire(self, source, target,
                  bend_out=False, bend_in=False, style=None):
        out = -90 if not bend_out or source[0] == target[0]\
            else (180 if source[0] > target[0] else 0)
        inp = 90 if not bend_in or source[0] == target[0]\
            else (180 if source[0] < target[0] else 0)
        looseness = 1
        if not (source[0] == target[0] or source[1] == target[1]):
            dx, dy = abs(source[0] - target[0]), abs(source[1] - target[1])
            length = sqrt(dx * dx + dy * dy)
            distance = min(dx, dy)
            looseness = round(distance / length * 2.1, 4)
        if looseness != 1:
            if style is None:
                style = ''
            style += f'looseness={looseness}'

        cmd = (
            "\\draw [in={}, out={}{}] "
            "({}.center) to ({}.center);\n")
        if source not in self.nodes:
            self.add_node(*source)
        if target not in self.nodes:
            self.add_node(*target)
        self.edgelayer.append(cmd.format(
            inp, out,
            f", {style}" if style is not None else "",
            self.nodes[source], self.nodes[target]))
        super().draw_wire(source, target, bend_out=bend_out, bend_in=bend_in)

    def draw_bezier(self, points):
        for point in points:
            if tuple(point) not in self.nodes:
                self.add_node(*point)
        self.edgelayer.append(
            "\\draw ({}.center) .. controls ({}.center) and ({}.center) .. "
            "({}.center);\n".format(*(self.nodes[tuple(p)] for p in points)))
        super().draw_bezier(points)

    def draw_filled_shape(self, start, steps, color):
        def node(point):
            if tuple(point) not in self.nodes:
                self.add_node(*point)
            return self.nodes[tuple(point)]
        path = f"({node(start)}.center)"
        for step in steps:
            if step[0] == "line":
                path += f" to ({node(step[1])}.center)"
            else:
                path += " .. controls ({}.center) and ({}.center) .. "\
                    "({}.center)".format(*(node(p) for p in step[1:]))
        self.edgelayer.append(
            f"\\draw [fill={self.format_color(color)}, draw=none] "
            f"{path} -- cycle;\n")
        super().draw_filled_shape(start, steps, color)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        spiders = [(node, node.box.color, node.box.shape)
                   for node in graph.nodes
                   if node.kind == "box" and node.box.draw_as_spider]
        for node, color, shape in spiders:
            i, j = graph.positions[node]
            text = node.box.drawing_name if draw_box_labels else ""
            if self.use_tikzstyles:
                style = f"\\tikzstyle{{{node.box.tikzstyle_name}}}=" \
                        f"[fill={self.format_color(color)}]\n"
                if style not in self.node_styles:
                    self.node_styles.append(style)
                options = f"style={node.box.tikzstyle_name}"
            else:
                options = f"{shape}, fill={color}"
            if params.get("nodesize", 1) != 1:
                options +=\
                    f", scale={params.get('nodesize')}"  # pragma: no cover
            self.add_node(i, j, text, options)
        super().draw_spiders(graph, draw_box_labels)

    def output(self, path=None, show=True, **params):
        baseline = params.get("baseline", 0)
        tikz_options = params.get("tikz_options", None)
        output_tikzstyle = self.use_tikzstyles\
            and params.get("output_tikzstyle", True)
        options = "baseline=(0.base)" if tikz_options is None\
            else "baseline=(0.base), " + tikz_options
        begin = [f"\\begin{{tikzpicture}}[{options}]\n"]
        nodes = ["\\begin{pgfonlayer}{nodelayer}\n",
                 f"\\node (0) at (0, {baseline}) {{}};\n"]\
            + self.nodelayer + ["\\end{pgfonlayer}\n"]
        edges = ["\\begin{pgfonlayer}{edgelayer}\n"] + self.edgelayer\
            + ["\\end{pgfonlayer}\n"]
        end = ["\\end{tikzpicture}\n"]
        if path is not None:
            if output_tikzstyle:
                style_path = '.'.join(path.split('.')[:-1]) + '.tikzstyles'
                with open(style_path, 'w+') as file:
                    file.writelines(["% Node styles\n"] + self.node_styles)
                    file.writelines(["% Edge styles\n"] + self.edge_styles)
            with open(path, 'w+') as file:
                file.writelines(begin + nodes + edges + end)
        elif show:  # pragma: no cover
            if output_tikzstyle:
                print(''.join(self.node_styles + self.edge_styles))
            print(''.join(begin + nodes + edges + end))


class Matplotlib(Backend):
    """ Matplotlib drawing backend. """
    def __init__(self, axis=None, figsize=None, linewidth=1):
        self.axis = axis or plt.subplots(figsize=figsize, facecolor='white')[1]
        self.linewidth = linewidth
        super().__init__()

    def draw_text(self, text, i, j, **params):
        params['fontsize'] = params.get('fontsize', DEFAULT['fontsize'])
        self.axis.text(i, j, text, **params)
        super().draw_text(text, i, j, **params)

    def draw_node(self, i, j, **params):
        self.axis.scatter(
            [i], [j],
            c=COLORS[params.get("color", "black")],
            marker=SHAPES[params.get("shape", "circle")],
            s=300 * params.get("nodesize", 1),
            edgecolors=params.get("edgecolor", None))
        super().draw_node(i, j, **params)

    def draw_polygon(
            self,
            *points,
            facecolor=DEFAULT["facecolor"],
            edgecolor=DEFAULT["edgecolor"]):
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        self.axis.add_patch(PathPatch(
            path,
            linewidth=self.linewidth,
            facecolor=COLORS[facecolor],
            edgecolor=COLORS[edgecolor]))
        super().draw_polygon(*points)

    def draw_wire(self, source, target,
                  bend_out=False, bend_in=False, style=None):
        if style == '->':  # pragma: no cover
            self.axis.arrow(
                *(source + (target[0] - source[0], target[1] - source[1])),
                head_width=.02, color="black")
        else:
            mid = (target[0], source[1])\
                if bend_out else (source[0], target[1])
            path = Path([source, mid, target],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            self.axis.add_patch(PathPatch(
                path, facecolor='none', linewidth=self.linewidth))
        super().draw_wire(source, target, bend_out=bend_out, bend_in=bend_in)

    def draw_bezier(self, points):
        path = Path(
            list(points),
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.axis.add_patch(PathPatch(
            path, facecolor='none', linewidth=self.linewidth))
        super().draw_bezier(points)

    def draw_filled_shape(self, start, steps, color):
        vertices, codes = [start], [Path.MOVETO]
        for step in steps:
            if step[0] == "line":
                vertices.append(step[1])
                codes.append(Path.LINETO)
            else:
                vertices += [step[1], step[2], step[3]]
                codes += 3 * [Path.CURVE4]
        vertices.append(start)
        codes.append(Path.CLOSEPOLY)
        self.axis.add_patch(PathPatch(
            Path(vertices, codes), facecolor=COLORS[color],
            edgecolor='none', linewidth=0))
        super().draw_filled_shape(start, steps, color)

    def draw_spiders(self, graph, draw_box_labels=True, **params):
        import networkx as nx
        nodes = {node for node in graph.nodes
                 if node.kind == "box" and node.box.draw_as_spider}
        shapes = {node: node.box.shape for node in nodes}
        for shape in set(shapes.values()):
            colors = {n: n.box.color for n, s in shapes.items() if s == shape}
            nodes, colors = zip(*colors.items())
            nx.draw_networkx_nodes(
                *graph.inside, nodelist=nodes,
                node_color=[COLORS[color] for color in colors],
                node_shape=SHAPES[shape], ax=self.axis,
                node_size=300 * params.get("nodesize", 1))
            if draw_box_labels:
                labels = {node: node.box.drawing_name for node in nodes}
                nx.draw_networkx_labels(*graph.inside, labels)
        super().draw_spiders(graph, draw_box_labels)

    def output(self, path=None, show=True, **params):
        xlim, ylim = params.get("xlim", None), params.get("ylim", None)
        margins = params.get("margins", DEFAULT['margins'])
        plt.margins(*margins)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis.set_aspect(params.get("aspect"))
        plt.axis('off')
        if xlim is not None:
            self.axis.set_xlim(*xlim)
        if ylim is not None:
            self.axis.set_ylim(*ylim)
        if path is not None:
            plt.savefig(path)
            plt.close()
        if show:
            plt.show()
