# -*- coding: utf-8 -*-
"""
DisCopy's legacy drawing: turn a diagram into a directed graph then plot it.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Node
    Backend
    TikzBackend
    MatBackend
    Equation

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        diagram2nx
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from math import sqrt
from tempfile import NamedTemporaryFile, TemporaryDirectory

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from discopy.config import (  # noqa: F401
    DRAWING_ATTRIBUTES as ATTRIBUTES,
    DRAWING_DEFAULT as DEFAULT, COLORS, SHAPES)

if TYPE_CHECKING:
    from discopy import monoidal


class Node:
    """ Node in a :class:`networkx.Graph`, can hold arbitrary data. """
    def __init__(self, kind, **data):
        self.kind, self.data = kind, data
        for key, value in data.items():
            setattr(self, key, value)

    def __eq__(self, other):
        return isinstance(other, Node)\
            and (self.kind, self.data) == (other.kind, other.data)

    def __repr__(self):
        str_data = ", ".join(
            f"{key}={repr(value)}"
            for key, value in sorted(self.data.items())
        )
        return f"Node({repr(self.kind)}, {str_data})"

    def __hash__(self):
        return hash(repr(self))

    __str__ = __repr__


def diagram2nx(diagram):
    """
    Builds a networkx graph, called by :meth:`Diagram.draw`.

    Parameters
    ----------
    diagram : discopy.monoidal.Diagram
        any diagram.

    Returns
    -------
    graph : networkx.DiGraph
        with nodes for inputs, outputs, boxes and their co/domain.

    positions : Mapping[Node, Tuple[float, float]]
        from nodes to pairs of floats.
    """
    import networkx as nx
    graph, pos = nx.DiGraph(), dict()

    def add_node(node, position):
        graph.add_node(node)
        pos.update({node: position})

    def add_box(scan, box, off, depth, x_pos):
        bubble_opening = getattr(box, "bubble_opening", False)
        bubble_closing = getattr(box, "bubble_closing", False)
        bubble = bubble_opening or bubble_closing
        node = Node("box", box=box, depth=depth)
        add_node(node, (x_pos, len(diagram) - depth - .5))
        for i, obj in enumerate(box.dom.inside):
            wire, position = Node("dom", obj=obj, i=i, depth=depth), (
                pos[scan[off + i]][0], len(diagram) - depth - .25)
            add_node(wire, position)
            graph.add_edge(scan[off + i], wire)
            if not bubble or bubble_closing and i in [0, len(box.dom) - 1]:
                graph.add_edge(wire, node)
        for i, obj in enumerate(box.cod.inside):
            position = (
                pos[scan[off + i]][0] if len(box.dom) == len(box.cod)
                else x_pos - len(box.cod[1:]) / 2 + i,
                len(diagram) - depth - .75)
            wire = Node("cod", obj=obj, i=i, depth=depth)
            add_node(wire, position)
            if not bubble or bubble_opening and i in [0, len(box.cod) - 1]:
                graph.add_edge(node, wire)
        if bubble_opening or bubble_closing:
            source_ty, target_ty = (box.dom, box.cod[1:-1]) if bubble_opening\
                else (box.dom[1:-1], box.cod)
            for i, (source_obj, target_obj) in enumerate(zip(
                    source_ty.inside, target_ty.inside)):
                source_i, target_i = (i, i + 1) if bubble_opening\
                    else (i + 1, i)
                source = Node("dom", obj=source_obj, i=source_i, depth=depth)
                target = Node("cod", obj=target_obj, i=target_i, depth=depth)
                graph.add_edge(source, target)
        return scan[:off]\
            + [Node("cod", obj=obj, i=i, depth=depth)
               for i, obj in enumerate(box.cod.inside)]\
            + scan[off + len(box.dom):]

    def make_space(scan, box, off):
        if not scan:
            return 0
        half_width = len(box.cod[:-1]) / 2 + 1
        if not box.dom:
            if not off:
                x_pos = pos[scan[0]][0] - half_width
            elif off == len(scan):
                x_pos = pos[scan[-1]][0] + half_width
            else:
                right = pos[scan[off + len(box.dom)]][0]
                x_pos = (pos[scan[off - 1]][0] + right) / 2
        else:
            right = pos[scan[off + len(box.dom) - 1]][0]
            x_pos = (pos[scan[off]][0] + right) / 2
        if off and pos[scan[off - 1]][0] > x_pos - half_width:
            limit = pos[scan[off - 1]][0]
            pad = limit - x_pos + half_width
            for node, position in pos.items():
                if position[0] <= limit:
                    pos[node] = (pos[node][0] - pad, pos[node][1])
        if off + len(box.dom) < len(scan)\
                and pos[scan[off + len(box.dom)]][0] < x_pos + half_width:
            limit = pos[scan[off + len(box.dom)]][0]
            pad = x_pos + half_width - limit
            for node, position in pos.items():
                if position[0] >= limit:
                    pos[node] = (pos[node][0] + pad, pos[node][1])
        return x_pos

    scan = []
    for i, obj in enumerate(diagram.dom.inside):
        node = Node("input", obj=obj, i=i)
        add_node(node, (i, len(diagram) or 1))
        scan.append(node)
    for depth, (box, off) in enumerate(zip(diagram.boxes, diagram.offsets)):
        x_pos = make_space(scan, box, off)
        scan = add_box(scan, box, off, depth, x_pos)
    for i, obj in enumerate(diagram.cod.inside):
        node = Node("output", obj=obj, i=i)
        add_node(node, (pos[scan[i]][0], 0))
        graph.add_edge(scan[i], node)
    return graph, pos


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

    def draw_polygon(self, *points, color=DEFAULT["color"]):
        """ Draws a polygon given a list of points. """
        self.max_width = max(self.max_width, max(i for i, _ in points))

    def draw_wire(self, source, target,
                  bend_out=False, bend_in=False, style=None):
        """ Draws a wire from source to target, possibly with a Bezier. """
        self.max_width = max(self.max_width, source[0], target[0])

    def draw_spiders(self, graph, positions, draw_box_labels=True, **params):
        """ Draws a list of boxes depicted as spiders. """
        spider_widths = [
            positions[n][0] for n in graph.nodes
            if n.kind == 'box' and n.box.draw_as_spider]
        if spider_widths:
            self.max_width = max(self.max_width, max(spider_widths))

    @abstractmethod
    def output(self, path=None, show=True, **params):
        """ Output the drawing. """


class TikzBackend(Backend):
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

    def draw_polygon(self, *points, color=DEFAULT["color"]):
        nodes = []
        for point in points:
            nodes.append(self.add_node(*point))
        nodes.append(nodes[0])
        if self.use_tikzstyles:
            style_name = "box" if color == DEFAULT["color"]\
                else f"{color}_box"
            style = f"\\tikzstyle{{{style_name}}}=" \
                    f"[-, fill={self.format_color(color)}]\n"
            if style not in self.edge_styles:
                self.edge_styles.append(style)
            options = f"style={style_name}"
        else:
            options = f"-, fill={{{color}}}"
        str_connections = " to ".join(f"({node}.center)" for node in nodes)
        self.edgelayer.append(f"\\draw [{options}] {str_connections};\n")
        super().draw_polygon(*points, color=color)

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

    def draw_spiders(self, graph, positions, draw_box_labels=True, **params):
        spiders = [(node, node.box.color, node.box.shape)
                   for node in graph.nodes
                   if node.kind == "box" and node.box.draw_as_spider]
        for node, color, shape in spiders:
            i, j = positions[node]
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
        super().draw_spiders(graph, positions, draw_box_labels)

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


class MatBackend(Backend):
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

    def draw_polygon(self, *points, color=DEFAULT["color"]):
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        self.axis.add_patch(PathPatch(
            path, facecolor=COLORS[color], linewidth=self.linewidth))
        super().draw_polygon(*points, color=color)

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

    def draw_spiders(self, graph, positions, draw_box_labels=True, **params):
        import networkx as nx
        nodes = {node for node in graph.nodes
                 if node.kind == "box" and node.box.draw_as_spider}
        shapes = {node: node.box.shape for node in nodes}
        for shape in set(shapes.values()):
            colors = {n: n.box.color for n, s in shapes.items() if s == shape}
            nodes, colors = zip(*colors.items())
            nx.draw_networkx_nodes(
                graph, positions, nodelist=nodes,
                node_color=[COLORS[color] for color in colors],
                node_shape=SHAPES[shape], ax=self.axis,
                node_size=300 * params.get("nodesize", 1))
            if draw_box_labels:
                labels = {node: node.box.drawing_name for node in nodes}
                nx.draw_networkx_labels(graph, positions, labels)
        super().draw_spiders(graph, positions, draw_box_labels)

    def output(self, path=None, show=True, **params):
        xlim, ylim = params.get("xlim", None), params.get("ylim", None)
        margins = params.get("margins", DEFAULT['margins'])
        aspect = params.get("aspect", DEFAULT['aspect'])
        plt.margins(*margins)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis.set_aspect(aspect)
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


def needs_asymmetry(diagram):
    if hasattr(diagram, "terms"):
        return any(needs_asymmetry(d) for d in diagram.terms)
    return any(
        box.is_dagger
        or getattr(box, "is_conjugate", False)
        or getattr(box, "is_transpose", False)
        for box in diagram.boxes)


def draw(diagram, **params):
    """
    Draws a diagram using networkx and matplotlib.

    Parameters
    ----------
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is :code:`False`.
    color : string, optional
        Color of the box or node, default is white (:code:`'#ffffff'`) for
        boxes and red (:code:`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.1)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is :code:`False`.
    draw_box_labels : bool, optional
        Whether to draw box labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    nodesize : float, optional
        Node size for spiders and controlled gates.
    fontsize : int, optional
        Font size for the boxes, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call :code:`plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.
    asymmetry : float, optional
        Make a box and its dagger mirror images, default is
        :code:`.25 * any(box.is_dagger for box in diagram.boxes)`.
    """
    diagram = diagram.to_drawing()

    drawing_methods = [
        ("draw_as_brakets", draw_brakets),
        ("draw_as_controlled", draw_controlled_gate),
        ("draw_as_discards", draw_discard),
        ("draw_as_measures", draw_measure),
        (None, draw_box)]

    params['asymmetry'] = params.get(
        'asymmetry', .25 * needs_asymmetry(diagram))

    def draw_wires(backend, graph, positions):
        for source, target in graph.edges():
            def inside_a_box(node):
                return node.kind == "box"\
                    and not node.box.draw_as_wires\
                    and not node.box.draw_as_spider
            if inside_a_box(source) or inside_a_box(target):
                continue  # no need to draw wires inside a box
            braid_shadow, source_position, target_position =\
                DEFAULT["braid_shadow"], positions[source], positions[target]
            bend_out, bend_in = source.kind == "box", target.kind == "box"
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
            backend.draw_wire(
                source_position, target_position, bend_out, bend_in)
            if source.kind in ["input", "cod"]\
                    and (params.get('draw_type_labels', True)
                         or getattr(source.obj, "always_draw_label", False)
                         and params.get('draw_box_labels', True)):
                i, j = positions[source]
                pad_i, pad_j = params.get('textpad', DEFAULT['textpad'])
                pad_j = 0 if source.kind == "input" else pad_j
                backend.draw_text(
                    str(source.obj), i + pad_i, j - pad_j,
                    fontsize=params.get('fontsize_types',
                                        params.get('fontsize', None)),
                    verticalalignment='top')
        return backend

    def scale_and_pad(graph, pos, scale, pad):
        if len(pos) == 0:
            return pos
        widths, heights = zip(*pos.values())
        min_width, min_height = min(widths), min(heights)
        pos = {n: ((x - min_width) * scale[0] + pad[0],
                   (y - min_height) * scale[1] + pad[1])
               for n, (x, y) in pos.items()}
        for box_node in graph.nodes:
            if box_node.kind == "box":
                for i, obj in enumerate(box_node.box.dom.inside):
                    node = Node("dom", obj=obj, i=i, depth=box_node.depth)
                    pos[node] = (
                        pos[node][0], pos[node][1] - .25 * (scale[1] - 1))
                for i, obj in enumerate(box_node.box.cod.inside):
                    node = Node("cod", obj=obj, i=i, depth=box_node.depth)
                    pos[node] = (
                        pos[node][0], pos[node][1] + .25 * (scale[1] - 1))
        return pos

    scale, pad = params.get('scale', (1, 1)), params.get('pad', (0, 0))
    graph, positions = diagram2nx(diagram)
    positions = scale_and_pad(graph, positions, scale, pad)
    backend = params.pop('backend') if 'backend' in params else\
        TikzBackend(use_tikzstyles=params.get('use_tikzstyles', None))\
        if params.get('to_tikz', False)\
        else MatBackend(figsize=params.get('figsize', None),
                        linewidth=params.get('linewidth', 1))

    min_size = 0.01
    max_v = max([v for p in positions.values() for v in p] + [min_size])
    params['nodesize'] = round(params.get('nodesize', 1.) / sqrt(max_v), 3)

    backend = draw_wires(backend, graph, positions)
    backend.draw_spiders(graph, positions, **params)
    box_nodes = [node for node in graph.nodes if node.kind == "box"]
    for node in box_nodes:
        if node.box.draw_as_spider or node.box.draw_as_wires:
            continue
        for attr, drawing_method in drawing_methods:
            if attr is None or getattr(node.box, attr, False):
                backend = drawing_method(backend, positions, node, **params)
                break
    return backend.output(
        path=params.get('path', None),
        baseline=len(box_nodes) / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT['margins']),
        aspect=params.get('aspect', DEFAULT['aspect']))


def draw_box(backend, positions, node, **params):
    """ Draws a box node on a given backend. """
    box, depth = node.box, node.depth
    asymmetry = params.get('asymmetry')
    if not box.dom and not box.cod:
        left, right = positions[node][0], positions[node][0]
    elif not box.dom:
        left, right = (
            positions[Node("cod", obj=box.cod.inside[i], i=i, depth=depth)][0]
            for i in [0, len(box.cod) - 1])
    elif not box.cod:
        left, right = (
            positions[Node("dom", obj=box.dom.inside[i], i=i, depth=depth)][0]
            for i in [0, len(box.dom) - 1])
    else:
        top_left, top_right = (
            positions[Node("dom", obj=box.dom.inside[i], i=i, depth=depth)][0]
            for i in [0, len(box.dom) - 1])
        bottom_left, bottom_right = (
            positions[Node("cod", obj=box.cod.inside[i], i=i, depth=depth)][0]
            for i in [0, len(box.cod) - 1])
        left = min(top_left, bottom_left)
        right = max(top_right, bottom_right)
    height = positions[node][1] - .25
    left, right = left - .25, right + .25

    # dictionary key is (is_dagger, is_conjugate)
    points = [[left, height], [right, height],
              [right, height + .5], [left, height + .5]]
    box.is_conjugate = getattr(box, "is_conjugate", False)
    box.is_transpose = getattr(box, "is_transpose", False)
    if box.is_transpose:
        points[0][0] -= asymmetry
    elif box.is_conjugate:
        points[3][0] -= asymmetry
    elif box.is_dagger:
        points[1][0] += asymmetry
    else:
        points[2][0] += asymmetry
    backend.draw_polygon(*points, color=box.color)
    if params.get('draw_box_labels', True):
        backend.draw_text(box.drawing_name, *positions[node],
                          ha='center', va='center',
                          fontsize=params.get('fontsize', None))
    return backend


def to_gif(diagram, *diagrams, **params):  # pragma: no cover
    """
    Builds a gif with the normalisation steps.

    Parameters
    ----------
    diagrams : :class:`Diagram`, optional
        Sequence of diagrams to draw.
    path : str
        Where to save the image, if :code:`None` a gif gets created.
    timestep : int, optional
        Time step in milliseconds, default is :code:`500`.
    loop : bool, optional
        Whether to loop, default is :code:`False`
    params : any, optional
        Passed to :meth:`Diagram.draw`.
    """
    path = params.pop("path", None)
    timestep = params.get("timestep", 500)
    loop = params.get("loop", False)
    steps, frames = (diagram, ) + diagrams, []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)
    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, f'{i}.png')
            _diagram.draw(path=tmp_path, **params)
            frames.append(Image.open(tmp_path))
        if loop:
            frames = frames + frames[::-1]
        frames[0].save(path, format='GIF', append_images=frames[1:],
                       save_all=True, duration=timestep,
                       **{'loop': 0} if loop else {})
        try:
            from IPython.display import HTML
            return HTML(f'<img src="{path}">')
        except ImportError:
            return f'<img src="{path}">'


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
    def __init__(self, *terms: monoidal.Diagram, symbol="=", space=1):
        self.terms, self.symbol, self.space = terms, symbol, space

    def __repr__(self):
        return f"Equation({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f" {self.symbol} ".join(map(str, self.terms))

    def draw(self, path=None, space=None, **params):
        """
        Drawing an equation.

        Parameters:
            path : Where to save the drawing.
            space : The amount of space between the terms.
            params : Passed to :meth:`discopy.monoidal.Diagram.draw`.
        """
        def height(term):
            # i.e. if isinstance(diagram, (Sum, Equation))
            if hasattr(term, "terms"):
                return max(height(d) for d in term.terms)
            return len(term.to_drawing()) or 1

        params['asymmetry'] = params.get(
            'asymmetry', .25 * needs_asymmetry(self))
        space = space or self.space
        max_height = max(map(height, self.terms))
        pad = params.get('pad', (0, 0))
        scale_x, scale_y = params.get('scale', (1, 1))
        backend = params['backend'] if 'backend' in params\
            else TikzBackend(
                use_tikzstyles=params.get('use_tikzstyles', None))\
            if params.get('to_tikz', False)\
            else MatBackend(figsize=params.get('figsize', None))

        for i, term in enumerate(self.terms):
            scale = (scale_x, scale_y * max_height / height(term))
            term.draw(**dict(
                params, show=False, path=None,
                backend=backend, scale=scale, pad=pad))
            pad = (backend.max_width + space, 0)
            if i < len(self.terms) - 1:
                backend.draw_text(
                    self.symbol, pad[0], scale_y * max_height / 2)
                pad = (pad[0] + space, pad[1])

        return backend.output(
            path=path,
            baseline=max_height / 2,
            tikz_options=params.get('tikz_options', None),
            show=params.get("show", True),
            margins=params.get('margins', DEFAULT['margins']),
            aspect=params.get('aspect', DEFAULT['aspect']))

    def __bool__(self):
        return all(term == self.terms[0] for term in self.terms)


def draw_discard(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.circuit.Discard` box. """
    box, depth = node.box, node.depth
    for i in range(len(box.dom)):
        obj = box.dom.inside[i]
        wire = Node("dom", obj=obj, depth=depth, i=i)
        middle = positions[wire]
        left, right = middle[0] - .25, middle[0] + .25
        height = positions[node][1] + .25
        for j in range(3):
            source = (left + .1 * j, height - .1 * j)
            target = (right - .1 * j, height - .1 * j)
            backend.draw_wire(source, target)

    return backend


def draw_measure(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.circuit.Measure` box. """
    backend = draw_box(backend, positions, node,
                       **dict(params, draw_box_labels=False))
    i, j = positions[node]
    backend.draw_wire((i - .15, j - .1), (i, j + .1), bend_in=True)
    backend.draw_wire((i, j + .1), (i + .15, j - .1), bend_out=True)
    backend.draw_wire((i, j - .1), (i + .05, j + .15), style='->')
    return backend


def draw_brakets(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.gates.Ket` box. """
    box, depth = node.box, node.depth
    is_bra = len(box.dom) > 0
    for i, bit in enumerate(box._digits):
        kind = "dom" if is_bra else "cod"
        obj = box.dom.inside[i] if is_bra else box.cod.inside[i]
        wire = Node(kind, obj=obj, depth=depth, i=i)
        middle = positions[wire]
        left = middle[0] - .25, middle[1]
        right = middle[0] + .25, middle[1]
        top = middle[0], middle[1] + .5
        bottom = middle[0], middle[1] - .5
        backend.draw_polygon(
            left, right, bottom if is_bra else top, color=box.color)
        backend.draw_text(
            bit, middle[0], middle[1] + (-.25 if is_bra else .2),
            ha='center', va='center', fontsize=params.get('fontsize', None))
    return backend


def draw_controlled_gate(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.gates.Controlled` gate. """
    box, depth = node.box, node.depth
    distance = box.distance
    c_size = len(box.controlled.dom)

    index = (0, distance) if distance > 0 else (c_size - distance - 1, 0)
    dom = Node("dom", obj=box.dom.inside[0], i=index[0], depth=depth)
    cod = Node("cod", obj=box.cod.inside[0], i=index[0], depth=depth)
    middle = positions[dom][0], (positions[dom][1] + positions[cod][1]) / 2
    controlled_box = box.controlled.to_drawing()
    controlled = Node("box", box=controlled_box, depth=depth)
    # TODO select obj properly for classical gates
    c_dom = Node("dom", obj=box.dom.inside[0], i=index[1], depth=depth)
    c_cod = Node("cod", obj=box.cod.inside[0], i=index[1], depth=depth)
    c_middle =\
        positions[c_dom][0], (positions[c_dom][1] + positions[c_cod][1]) / 2
    target = (positions[c_dom][0] + (c_size - 1) / 2,
              (positions[c_dom][1] + positions[c_cod][1]) / 2)
    target_boundary = target
    if controlled_box.name == "X":  # CX gets drawn as a circled plus sign.
        backend.draw_wire(positions[c_dom], positions[c_cod])
        eps = 1e-10
        perturbed_target = target[0], target[1] + eps
        backend.draw_node(
            *perturbed_target,
            shape="circle", color="white", edgecolor="black",
            nodesize=2 * params.get("nodesize", 1))
        backend.draw_node(
            *target, shape="plus",
            nodesize=2 * params.get("nodesize", 1))
    else:
        fake_positions = {controlled: target}
        for i in range(c_size):
            dom_node = Node("dom", obj=box.dom.inside[i], i=i, depth=depth)
            x, y = positions[c_dom][0] + i, positions[c_dom][1]
            fake_positions[dom_node] = x, y

            cod_node = Node("cod", obj=box.cod.inside[i], i=i, depth=depth)
            x, y = positions[c_cod][0] + i, positions[c_cod][1]
            fake_positions[cod_node] = x, y

        shift_boundary = True
        if hasattr(box.controlled, "draw_as_controlled"):
            backend = draw_controlled_gate(
                backend, fake_positions, controlled, **params)

            next_box = box.controlled
            while hasattr(next_box, "controlled"):
                if controlled_box.distance * next_box.distance < 0:
                    shift_boundary = False
                    break
                next_box = next_box.controlled
            if next_box.name == "X":
                shift_boundary = False
        else:
            backend = draw_box(backend, fake_positions, controlled, **params)

        if shift_boundary:
            if box.distance > 0:
                target_boundary = c_middle[0] - .25, c_middle[1]
            else:
                target_boundary = c_middle[0] + c_size - 1 + .25, c_middle[1]
        else:
            if box.distance > 0:
                target_boundary = c_middle[0], c_middle[1]
            else:
                target_boundary = c_middle[0] + c_size - 1, c_middle[1]
    backend.draw_wire(positions[dom], positions[cod])

    # draw all the other vertical wires
    extra_offset = 1 if distance > 0 else len(box.controlled.dom)
    for i in range(extra_offset, extra_offset + abs(distance) - 1):
        node1 = Node("dom", obj=box.dom.inside[i], i=i, depth=depth)
        node2 = Node("cod", obj=box.cod.inside[i], i=i, depth=depth)
        backend.draw_wire(positions[node1], positions[node2])

    # TODO change bend_in and bend_out for tikz backend
    backend.draw_wire(middle, target_boundary, bend_in=True, bend_out=True)

    backend.draw_node(
        *middle, color="black", shape="circle",
        nodesize=params.get("nodesize", 1))
    return backend
