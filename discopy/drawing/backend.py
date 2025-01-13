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
    backend.draw_wires(graph, **params)
    backend.draw_boxes(graph, **params)
    backend.draw_spiders(graph, **params)

    return backend.output(
        path=params.get('path', None),
        baseline=graph.height / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True), aspect=aspect,
        margins=params.get('margins', DEFAULT['margins']))


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
        self.draw_text(label, i, j, verticalalignment='top', fontsize=fontsize)

    def draw_wires(self, graph, **params):
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
            bend_out, bend_in = source.kind == "box", target.kind == "box"
            braid_shadow = DEFAULT["braid_shadow"]
            if source.kind == "box" and source.box.draw_as_braid:
                if source.box.is_dagger and target.i == 0:
                    source_position = tuple(
                        x + b * shadow
                        for x, b, shadow in zip(
                            source_position, [-1, -1], braid_shadow))
                if not source.box.is_dagger and target.i == 1:
                    source_position = tuple(
                        x + b * shadow
                        for x, b, shadow in zip(
                            source_position, [1, -1], braid_shadow))
            if target.kind == "box" and target.box.draw_as_braid:
                if target.box.is_dagger and source.i == 1:
                    target_position = tuple(
                        x + b * shadow
                        for x, b, shadow in zip(
                            target_position, [1, 1], braid_shadow))
                if not target.box.is_dagger and source.i == 0:
                    target_position = tuple(
                        x + b * shadow
                        for x, b, shadow in zip(
                            target_position, [-1, 1], braid_shadow))
            self.draw_wire(
                source_position, target_position, bend_out, bend_in)

    def draw_boxes(self, graph, **params):
        drawing_methods = [
            ("draw_as_brakets", "draw_brakets"),
            ("draw_as_controlled", "draw_controlled_gate"),
            ("draw_as_discards", "draw_discard"),
            ("draw_as_measures", "draw_measure"),
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
                           ha='center', va='center',
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
