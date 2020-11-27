# -*- coding: utf-8 -*-
"""
Drawing module.
"""

import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile, TemporaryDirectory

import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


@dataclass
class DEFAULT:
    """ Drawing defaults. """
    aspect = "auto"
    fontsize = 12
    margins = (.05, .05)
    textpad = (.1, .1)
    color = 'white'
    use_tikzstyles = False


@dataclass
class COLORS:
    """ Drawing colours. """
    white = '#ffffff'
    red = '#e8a5a5'
    green = '#d8f8d8'
    blue = '#776ff3'
    yellow = '#f7f700'
    black = '#000000'


@dataclass
class SHAPES:
    """ Drawing shapes. """
    rectangle = 's'
    circle = 'o'


class Node:
    """ Node in a :class:`networkx.Graph`, can hold arbitrary data. """
    def __init__(self, *data):
        self.data = data

    def __eq__(self, other):
        return isinstance(other, Node) and self.data == other.data\
            and type(self).__name__ == type(other).__name__

    def __hash__(self):
        return hash((type(self).__name__, ) + self.data)

    def __repr__(self):
        return "{}{}".format(type(self).__name__, self.data)

    __str__ = __repr__


class BoxNode(Node):
    """ Node in a networkx Graph, representing a box in a diagram. """
    def __init__(self, box, depth):
        super().__init__(box, depth)
        self.box, self.depth = box, depth


class WireNode(Node):
    """ Node in a networkx Graph, representing a wire in a diagram. """
    def __init__(self, obj, i, depth=None):
        super().__init__(obj, i, depth)
        self.obj, self.i, self.depth = obj, i, depth


class InputNode(WireNode):
    """ Node in a networkx Graph, representing an input wire in a diagram. """


class OutputNode(WireNode):
    """ Node in a networkx Graph, representing an output wire in a diagram. """


class DomNode(WireNode):
    """ Node in a networkx Graph, representing an input wire of a box. """


class CodNode(WireNode):
    """ Node in a networkx Graph, representing an output wire of a box. """


def diagram_to_nx(diagram):
    """
    Builds a networkx graph, called by :meth:`Diagram.draw`.

    Parameters
    ----------
    diagram : discopy.monoidal.Diagram
        any diagram.

    Returns
    -------
    graph : networkx.Graph
        with nodes for inputs, outputs, boxes and wires.

    positions : Mapping[Node, Tuple[float, float]]
        from nodes to pairs of floats.
    """
    graph, pos = nx.DiGraph(), dict()

    def add_node(node, position):
        graph.add_node(node)
        pos.update({node: position})

    def add_box(scan, box, off, depth, x_pos):
        node = BoxNode(box, depth)
        add_node(node, (x_pos, len(diagram) - depth - .5))
        for i, obj in enumerate(box.dom):
            wire, position = DomNode(obj, i, depth), (
                pos[scan[off + i]][0], len(diagram) - depth - .25)
            add_node(wire, position)
            graph.add_edge(scan[off + i], wire)
            graph.add_edge(wire, node)
        for i, obj in enumerate(box.cod):
            wire, position = CodNode(obj, i, depth), (
                x_pos - len(box.cod[1:]) / 2 + i, len(diagram) - depth - .75)
            add_node(wire, position)
            graph.add_edge(node, wire)
        return scan[:off]\
            + [CodNode(obj, i, depth) for i, obj in enumerate(box.cod)]\
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
    for i, obj in enumerate(diagram.dom):
        add_node(InputNode(obj, i), (i, len(diagram) or 1))
    scan = [InputNode(obj, i) for i, obj in enumerate(diagram.dom)]
    for depth, (box, off) in enumerate(zip(diagram.boxes, diagram.offsets)):
        x_pos = make_space(scan, box, off)
        scan = add_box(scan, box, off, depth, x_pos)
    for i, obj in enumerate(diagram.cod):
        add_node(OutputNode(obj, i), (pos[scan[i]][0], 0))
        graph.add_edge(scan[i], OutputNode(obj, i))
    return graph, pos


class Backend(ABC):
    """ Abstract drawing backend. """
    @abstractmethod
    def draw_text(self, text, i, j, **params):
        """ Draws a piece of text at a given position. """

    @abstractmethod
    def draw_polygon(self, *points, color=DEFAULT.color):
        """ Draws a polygon given a list of points. """

    @abstractmethod
    def draw_wire(self, source, target, bend_out=False, bend_in=False):
        """ Draws a wire from source to target, possibly with a Bezier. """

    @abstractmethod
    def draw_spiders(self, spiders, graph, positions, draw_box_labels=True):
        """ Draws a list of boxes depicted as spiders. """

    @abstractmethod
    def output(self, path=None, show=True, **params):
        """ Output the drawing. """


class TikzBackend(Backend):
    """ Tikz drawing backend. """
    def __init__(self, use_tikzstyles=None):
        self.use_tikzstyles = DEFAULT.use_tikzstyles\
            if use_tikzstyles is None else use_tikzstyles
        self.node_styles, self.edge_styles = [], []
        self.nodes, self.nodelayer, self.edgelayer = {}, [], []

    @staticmethod
    def format_color(color):
        hexcode = getattr(COLORS, color)
        rgb = [
            int(hex, 16) for hex in [hexcode[1:3], hexcode[3:5], hexcode[5:]]]
        return "{{rgb,255: red,{}; green,{}; blue,{}}}".format(*rgb)

    def add_node(self, i, j, text=None, options=None):
        """ Add a node to the tikz picture, return its unique id. """
        node = len(self.nodes) + 1
        self.nodelayer.append(
            "\\node [{}] ({}) at ({}, {}) {{{}}};\n".format(
                options or "", node, i, j, text or ""))
        self.nodes.update({(i, j): node})
        return node

    def draw_text(self, text, i, j, **params):
        options = "style=none"
        if params.get("verticalalignment", "center") == "top":  # wire labels
            options += ", right"
        if 'fontsize' in params and params['fontsize'] is not None:
            options += ", scale={}".format(params['fontsize'])
        self.add_node(i, j, text, options)

    def draw_polygon(self, *points, color=DEFAULT.color):
        nodes = []
        for point in points:
            nodes.append(self.add_node(*point))
        nodes.append(nodes[0])
        if self.use_tikzstyles:
            style_name = "box" if color == DEFAULT.color\
                else "{}_box".format(color)
            style = "\\tikzstyle{{{}}}=[-, fill={}]\n"\
                .format(style_name, self.format_color(color))
            if style not in self.edge_styles:
                self.edge_styles.append(style)
            options = "style={}".format(style_name)
        else:
            options = "-, fill={{{}}}".format(color)
        self.edgelayer.append("\\draw [{}] {};\n".format(options, " to ".join(
            "({}.center)".format(node) for node in nodes)))

    def draw_wire(self, source, target, bend_out=False, bend_in=False):
        out = -90 if not bend_out or source[0] == target[0]\
            else (180 if source[0] > target[0] else 0)
        inp = 90 if not bend_in or source[0] == target[0]\
            else (180 if source[0] < target[0] else 0)
        cmd = "\\draw [in={}, out={}] ({}.center) to ({}.center);\n"
        if source not in self.nodes:
            self.add_node(*source)
        if target not in self.nodes:
            self.add_node(*target)
        self.edgelayer.append(cmd.format(
            inp, out, self.nodes[source], self.nodes[target]))

    def draw_spiders(self, spiders, graph, positions, draw_box_labels=True):
        for node, color, shape in spiders:
            i, j = positions[node]
            text = getattr(node.box, "drawing_name", str(node.box))\
                if draw_box_labels else ""
            if self.use_tikzstyles:
                style_name = getattr(node.box, "tikzstyle_name", str(node.box))
                style = "\\tikzstyle{{{}}}=[fill={}]\n"\
                    .format(style_name, self.format_color(color))
                if style not in self.node_styles:
                    self.node_styles.append(style)
                options = "style={}".format(style_name)
            else:
                options = "{}, fill={}".format(shape, color)
            self.add_node(i, j, text, options)

    def output(self, path=None, show=True, **params):
        baseline = params.get("baseline", 0)
        tikz_options = params.get("tikz_options", None)
        output_tikzstyle = self.use_tikzstyles\
            and params.get("output_tikzstyle", True)
        options = "baseline=(0.base)" if tikz_options is None\
            else "baseline=(0.base), " + tikz_options
        begin = ["\\begin{{tikzpicture}}[{}]\n".format(options)]
        nodes = ["\\begin{pgfonlayer}{nodelayer}\n",
                 "\\node (0) at (0, {}) {{}};\n".format(baseline)]\
            + self.nodelayer + ["\\end{pgfonlayer}\n"]
        edges = ["\\begin{pgfonlayer}{edgelayer}\n"] + self.edgelayer +\
                ["\\end{pgfonlayer}\n"]
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
    def __init__(self, axis=None, figsize=None):
        self.axis = axis or plt.subplots(figsize=figsize)[1]

    def draw_text(self, text, i, j, **params):
        params['fontsize'] = params.get('fontsize', None) or DEFAULT.fontsize
        self.axis.text(i, j, text, **params)

    def draw_polygon(self, *points, color=DEFAULT.color):
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        self.axis.add_patch(PathPatch(path, facecolor=getattr(COLORS, color)))

    def draw_wire(self, source, target, bend_out=False, bend_in=False):
        mid = (target[0], source[1]) if bend_out else (source[0], target[1])
        path = Path([source, mid, target],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        self.axis.add_patch(PathPatch(path, facecolor='none'))

    def draw_spiders(self, spiders, graph, positions, draw_box_labels=True):
        shapes = {shape for _, _, shape in spiders}
        for shape in shapes:
            shaped_spiders = [
                (node, color) for node, color, s in spiders if s == shape]
            nodes, colors = zip(*shaped_spiders)
            hex_codes = [getattr(COLORS, color) for color in colors]
            nx.draw_networkx_nodes(
                graph, positions, nodelist=nodes,
                node_color=hex_codes,
                node_shape=getattr(SHAPES, shape), ax=self.axis)
            if draw_box_labels:
                labels = {n: getattr(n.box, "drawing_name", str(n.box))
                          for n in nodes}
                nx.draw_networkx_labels(graph, positions, labels)

    def output(self, path=None, show=True, **params):
        xlim, ylim = params.get("xlim", None), params.get("ylim", None)
        margins = params.get("margins", DEFAULT.margins)
        aspect = params.get("aspect", DEFAULT.aspect)
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


def draw(diagram, backend=None, data=None, **params):
    """ Draws a diagram, see :meth:`monoidal.Diagram.draw`. """
    asymmetry = params.get('asymmetry',
                           .25 * any(box.is_dagger for box in diagram.boxes))
    graph, positions = diagram_to_nx(diagram) if data is None else data
    spiders = [(BoxNode(box, depth),
                getattr(box, 'color', 'red'),
                getattr(box, 'shape', 'circle'))
               for depth, box in enumerate(diagram.boxes)
               if getattr(box, "draw_as_spider", False)]

    backend = backend if backend is not None else\
        TikzBackend(use_tikzstyles=params.get('use_tikzstyles', None))\
        if params.get('to_tikz', False)\
        else MatBackend(figsize=params.get('figsize', None))

    def draw_box(box, depth):
        node = BoxNode(box, depth)
        if getattr(box, "draw_as_wire", False):
            return
        if not box.dom and not box.cod:
            left, right = positions[node][0], positions[node][0]
        elif not box.dom:
            left, right = (
                positions[CodNode(box.cod[i], i, depth)][0]
                for i in [0, len(box.cod) - 1])
        elif not box.cod:
            left, right = (
                positions[DomNode(box.dom[i], i, depth)][0]
                for i in [0, len(box.dom) - 1])
        else:
            top_left, top_right = (
                positions[DomNode(box.dom[i], i, depth)][0]
                for i in [0, len(box.dom) - 1])
            bottom_left, bottom_right = (
                positions[CodNode(box.cod[i], i, depth)][0]
                for i in [0, len(box.cod) - 1])
            left = min(top_left, bottom_left)
            right = max(top_right, bottom_right)
        height = positions[node][1] - .25
        left, right = left - .25, right + .25
        backend.draw_polygon(
            (left, height),
            (right + (asymmetry if box.is_dagger else 0), height),
            (right + (0 if box.is_dagger else asymmetry), height + .5),
            (left, height + .5),
            color=params.get('color', DEFAULT.color))
        if params.get('draw_box_labels', True):
            label = getattr(box, "drawing_name", str(box))
            backend.draw_text(label, *positions[node],
                              ha='center', va='center',
                              fontsize=params.get('fontsize', None))

    def draw_wires():
        for source, target in graph.edges():
            def inside_a_box(node):
                return isinstance(node, BoxNode)\
                    and not getattr(node.box, "draw_as_wire", False)\
                    and not getattr(node.box, "draw_as_spider", False)
            if inside_a_box(source) or inside_a_box(target):
                continue  # no need to draw wires inside a box
            backend.draw_wire(
                positions[source], positions[target],
                bend_out=isinstance(source, BoxNode),
                bend_in=isinstance(target, BoxNode))
            if isinstance(source, (InputNode, CodNode))\
                    and params.get('draw_types', True):
                i, j = positions[source]
                pad_i, pad_j = params.get('textpad', DEFAULT.textpad)
                pad_j = 0 if isinstance(source, InputNode) else pad_j
                backend.draw_text(
                    str(source.obj), i + pad_i, j - pad_j,
                    fontsize=params.get('fontsize_types',
                                        params.get('fontsize', None)),
                    verticalalignment='top')

    draw_wires()
    backend.draw_spiders(
        spiders, graph, positions,
        draw_box_labels=params.get('draw_box_labels', True))
    for depth, box in enumerate(diagram.boxes):
        if getattr(box, "draw_as_spider", False):
            continue
        draw_box(box, depth)
    return backend.output(
        path=params.get('path', None),
        baseline=len(diagram) / 2 or .5,
        tikz_options=params.get('tikz_options', None),
        show=params.get('show', True),
        margins=params.get('margins', DEFAULT.margins),
        aspect=params.get('aspect', DEFAULT.aspect))


def to_gif(diagram, *diagrams, **params):  # pragma: no cover
    """ Draws a sequence of diagrams as an animated picture. """
    path = params.get("path", None)
    timestep = params.get("timestep", 500)
    loop = params.get("loop", False)
    steps, frames = (diagram, ) + diagrams, []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)
    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, '{}.png'.format(i))
            _diagram.draw(path=tmp_path, **params)
            frames.append(Image.open(tmp_path))
        if loop:
            frames = frames + frames[::-1]
        frames[0].save(path, format='GIF', append_images=frames[1:],
                       save_all=True, duration=timestep,
                       **{'loop': 0} if loop else {})
        try:
            from IPython.display import HTML
            return HTML('<img src="{}">'.format(path))
        except ImportError:
            return '<img src="{}">'.format(path)


def pregroup_draw(words, cups, **params):
    """
    Draws pregroup words and cups.
    """
    textpad = params.get('textpad', (.1, .2))
    textpad_words = params.get('textpad_words', (0, .1))
    space = params.get('space', .5)
    width = params.get('width', 2.)
    fontsize = params.get('fontsize', None)

    backend = TikzBackend(use_tikzstyles=params.get('use_tikzstyles', None))\
        if params.get('to_tikz', False)\
        else MatBackend(figsize=params.get('figsize', None))

    def draw_triangles(words):
        scan = []
        for i, word in enumerate(words.boxes):
            for j, _ in enumerate(word.cod):
                x_wire = (space + width) * i\
                    + (width / (len(word.cod) + 1)) * (j + 1)
                scan.append(x_wire)
                if params.get('draw_types', True):
                    backend.draw_text(
                        str(word.cod[j]), x_wire + textpad[0], -textpad[1],
                        fontsize=params.get('fontsize_types', fontsize))
            backend.draw_polygon(
                ((space + width) * i, 0),
                ((space + width) * i + width, 0),
                ((space + width) * i + width / 2, 1),
                color=DEFAULT.color)
            backend.draw_text(
                str(word), (space + width) * i + width / 2 + textpad_words[0],
                textpad_words[1], ha='center', fontsize=fontsize)
        return scan

    def draw_cups_and_wires(cups, scan):
        for j, off in [(j, off)
                       for j, s in enumerate(cups) for off in s.offsets]:
            middle = (scan[off] + scan[off + 1]) / 2
            backend.draw_wire((scan[off], 0), (middle, - j - 1), bend_in=True)
            backend.draw_wire(
                (scan[off + 1], 0), (middle, - j - 1), bend_in=True)
            scan = scan[:off] + scan[off + 2:]
        for i, _ in enumerate(cups[-1].cod if cups else words.cod):
            label = str(cups[-1].cod[i]) if cups else ""
            backend.draw_wire((scan[i], 0), (scan[i], - (len(cups) or 1) - 1))
            if params.get('draw_types', True):
                backend.draw_text(
                    label, scan[i] + textpad[0], - (len(cups) or 1) - space,
                    fontsize=params.get('fontsize_types', fontsize))

    scan = draw_triangles(words.normal_form())
    draw_cups_and_wires(cups, scan)
    backend.output(
        params.get('path', None),
        tikz_options=params.get('tikz_options', None),
        xlim=(0, (space + width) * len(words.boxes) - space),
        ylim=(- len(cups) - space, 1),
        margins=params.get('margins', DEFAULT.margins),
        aspect=params.get('aspect', DEFAULT.aspect))


def equation(*diagrams, path=None, symbol="=", space=1, **params):
    """ Draws an equation with multiple diagrams. """
    pad, max_height = 0, max(map(len, diagrams))
    scale_x, scale_y = params.get('scale', (1, 1))
    backend = TikzBackend(use_tikzstyles=params.get('use_tikzstyles', None))\
        if params.get('to_tikz', False)\
        else MatBackend(figsize=params.get('figsize', None))

    def scale_and_pad(diagram, pos, scale, pad):
        widths, heights = zip(*pos.values())
        min_width, min_height = min(widths), min(heights)
        pos = {n: ((x - min_width) * scale[0] + pad[0],
                   (y - min_height) * scale[1] + pad[1])
               for n, (x, y) in pos.items()}
        for depth, box in enumerate(diagram.boxes):
            for i, obj in enumerate(box.dom):
                node = DomNode(obj, i, depth)
                pos[node] = (
                    pos[node][0], pos[node][1] - .25 * (scale[1] - 1))
            for i, obj in enumerate(box.cod):
                node = CodNode(obj, i, depth)
                pos[node] = (
                    pos[node][0], pos[node][1] + .25 * (scale[1] - 1))
        return pos

    for i, diagram in enumerate(diagrams):
        scale = (scale_x, scale_y * max_height / (len(diagram) or 1))
        graph, positions = diagram_to_nx(diagram)
        positions = scale_and_pad(diagram, positions, scale, (pad, 0))
        diagram.draw(backend=backend, data=(graph, positions),
                     **dict(params, show=False, path=None))
        widths = {x for x, _ in positions.values()}
        min_width, max_width = min(widths), max(widths)
        pad += max_width - min_width + space
        if i < len(diagrams) - 1:
            backend.draw_text(symbol, pad, scale_y * max_height / 2)
            pad += space

    return backend.output(
        path=path,
        baseline=max_height / 2,
        tikz_options=params.get('tikz_options', None),
        show=params.get("show", True),
        margins=params.get('margins', DEFAULT.margins),
        aspect=params.get('aspect', DEFAULT.aspect))
