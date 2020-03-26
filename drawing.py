# -*- coding: utf-8 -*-
"""
Drawing module.
"""

import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import networkx as nx
from PIL import Image
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


WIRE_BOXES = ['CUP', 'CAP', 'SWAP']


def diagram_to_nx(diagram):
    """
    Builds a networkx graph, called by :meth:`Diagram.draw`.

    Returns
    -------
    graph, positions, labels : tuple
        where:

        * :code:`graph` is a networkx graph with nodes for inputs, outputs,
          boxes and wires,
        * :code:`positions` is a dict from nodes to pairs of floats,
        * :code:`labels` is a dict from nodes to strings.
    """
    graph, pos, labels = nx.DiGraph(), dict(), dict()

    def add_node(node, position, label=None):
        graph.add_node(node)
        pos.update({node: position})
        if label is not None:
            labels.update({node: label})

    def add_box(scan, box, off, depth, x_pos):
        node = 'wire_box_{}'.format(depth) if box.name in WIRE_BOXES\
            else 'box_{}'.format(depth)
        add_node(node,
                 (x_pos, len(diagram) - depth - .5), str(box))
        for i, _ in enumerate(box.dom):
            wire, position = 'wire_dom_{}_{}'.format(depth, i), (
                pos[scan[off + i]][0], len(diagram) - depth - .25)
            add_node(wire, position, str(box.dom[i]))
            graph.add_edge(scan[off + i], wire)
            graph.add_edge(wire, node)
        for i, _ in enumerate(box.cod):
            wire, position = 'wire_cod_{}_{}'.format(depth, i), (
                x_pos - len(box.cod[1:]) / 2 + i, len(diagram) - depth - .75)
            add_node(wire, position, str(box.cod[i]))
            graph.add_edge(node, wire)
        return scan[:off] + ['wire_cod_{}_{}'.format(depth, i)
                             for i, _ in enumerate(box.cod)]\
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

    for i, _ in enumerate(diagram.dom):
        add_node('input_{}'.format(i),
                 (i, len(diagram.boxes[:-1]) + 1), str(diagram.dom[i]))
    scan = ['input_{}'.format(i) for i, _ in enumerate(diagram.dom)]
    for depth, (box, off) in enumerate(zip(diagram.boxes, diagram.offsets)):
        x_pos = make_space(scan, box, off)
        scan = add_box(scan, box, off, depth, x_pos)
    for i, _ in enumerate(diagram.cod):
        add_node('output_{}'.format(i),
                 (pos[scan[i]][0], 0), str(diagram.cod[i]))
        graph.add_edge(scan[i], 'output_{}'.format(i))
    return graph, pos, labels


def save_tikz(commands, path=None):
    """
    Save a list of tikz commands.
    """
    with open(path, 'w+') as file:
        file.writelines(
            ["\\begin{tikzpicture}\n"] + commands + ["\\end{tikzpicture}\n"])


def draw_text(axis, text, i, j, to_tikz=False, **params):
    """
    Draws `text` on `axis` as position `(i, j)`.
    If `to_tikz`, axis is a list of tikz commands, else it's a matplotlib axis.
    `params` get passed to matplotlib.
    """
    if to_tikz:
        axis.append("\\node () at ({}, {}) {{{}}};\n".format(i, j, text))
    else:
        axis.text(i, j, text, **params)


def draw_polygon(axis, *points, to_tikz=False, color='#ffffff'):
    """
    Draws a polygon from a list of points.
    """
    if to_tikz:
        axis.append("\\draw {};\n".format(" -- ".join(
            "({}, {})".format(*x) for x in points + points[:1])))
    else:
        codes = [Path.MOVETO]
        codes += len(points[1:]) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(points + points[:1], codes)
        axis.add_patch(PathPatch(path, facecolor=color))


def draw_wire(axis, source, target,
              bend_out=False, bend_in=False, to_tikz=False):
    """
    Draws a wire from source to target using a Bezier curve.
    """
    if to_tikz:
        out = -90 if not bend_out or source[0] == target[0]\
            else (180 if source[0] > target[0] else 0)
        inp = 90 if not bend_in or source[0] == target[0]\
            else (180 if source[0] < target[0] else 0)
        cmd = "\\draw [out={}, in={}] {{}} to {{}};\n".format(out, inp)
        axis.append(cmd.format(*("({}, {})".format(*point)
                                 for point in [source, target])))
    else:
        mid = (target[0], source[1]) if bend_out else (source[0], target[1])
        path = Path([source, mid, target],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        axis.add_patch(PathPatch(path, facecolor='none'))


def draw(diagram, axis=None, data=None, **params):
    """
    Draws a diagram, see :meth:`moncat.Diagram.draw` for a list of parameters.
    """
    asymmetry = params.get('asymmetry',
                           .25 * any(box.is_dagger for box in diagram.boxes))
    graph, positions, labels = diagram_to_nx(diagram) if data is None else data

    scale_x, scale_y = params.get('scale', (1, 1))
    pad_x, pad_y = params.get('pad', (0, 0))
    positions = {n: (x * scale_x + pad_x, y * scale_y + pad_y)
                 for n, (x, y) in positions.items()}

    def draw_nodes(axis, nodes):
        if params.get('to_tikz', False):
            dec = "[circle, fill={}]".format(params.get('color', 'red'))
            cmd = "\\node {1} () at ({2}, {3}) {{{0}}};\n"
            for node in nodes:
                lab = labels[node]\
                    if params.get('draw_box_labels', True) else ""
                axis.append(cmd.format(lab, dec, *positions[node]))
        else:
            nx.draw_networkx_nodes(
                graph, positions, nodelist=nodes,
                node_color=params.get('color', '#ff0000'), ax=axis)
            if params.get('draw_box_labels', True):
                nx.draw_networkx_labels(
                    graph, positions,
                    {n: l for n, l in labels.items() if n in nodes})

    def draw_box(axis, box, depth):
        node = 'box_{}'.format(depth)
        if node not in graph.nodes():
            return
        if not box.dom and not box.cod:
            left, right = positions[node][0], positions[node][0]
        elif not box.dom:
            left, right = (
                positions['wire_cod_{}_{}'.format(depth, i)][0]
                for i in [0, len(box.cod) - 1])
        elif not box.cod:
            left, right = (
                positions['wire_dom_{}_{}'.format(depth, i)][0]
                for i in [0, len(box.dom) - 1])
        else:
            top_left, top_right = (
                positions['wire_dom_{}_{}'.format(depth, i)][0]
                for i in [0, len(box.dom) - 1])
            bottom_left, bottom_right = (
                positions['wire_cod_{}_{}'.format(depth, i)][0]
                for i in [0, len(box.cod) - 1])
            left = min(top_left, bottom_left)
            right = max(top_right, bottom_right)
        height = positions[node][1] - .25
        left, right = left - .25, right + .25
        draw_polygon(
            axis, (left, height),
            (right + (asymmetry if box.is_dagger else 0), height),
            (right + (0 if box.is_dagger else asymmetry), height + .5),
            (left, height + .5),
            to_tikz=params.get('to_tikz', False),
            color=params.get('color', '#ffffff'))
        if params.get('draw_box_labels', True):
            draw_text(axis, str(box.name), *positions[node],
                      to_tikz=params.get('to_tikz', False),
                      ha='center', va='center',
                      fontsize=params.get('fontsize', 12))

    def draw_wires(axis):
        for case in ['input', 'wire_cod']:
            for node in [n for n in graph.nodes if n[:len(case)] == case]:
                i, j = positions[node]
                if params.get('draw_types', True):
                    if node in labels.keys():
                        pad_i, pad_j = params.get('textpad', (.1, .1))
                        draw_text(
                            axis, labels[node],
                            i + pad_i, j - (0 if case == 'input' else pad_j),
                            to_tikz=params.get('to_tikz', False),
                            fontsize=params.get('fontsize_types',
                                                params.get('fontsize', 12)),
                            verticalalignment='top')
        for source, target in graph.edges():
            if "box" in (source[:3], target[:3])\
                    and not params.get('draw_as_nodes', False):
                continue
            draw_wire(axis, positions[source], positions[target],
                      bend_out='box' in source, bend_in='box' in target,
                      to_tikz=params.get('to_tikz', False))
    if axis is None:
        axis = [] if params.get('to_tikz', False)\
            else plt.subplots(figsize=params.get('figsize', None))[1]
    draw_wires(axis)
    if params.get('draw_as_nodes', False):
        draw_nodes(axis, [node for node in graph.nodes if node[:3] == 'box'])
    else:
        for depth, box in enumerate(diagram.boxes):
            draw_box(axis, box, depth)
    if params.get('to_tikz', False):
        if 'path' in params:
            save_tikz(axis, params['path'])
    else:
        plt.margins(*params.get('margins', (.05, .05)))
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        axis.set_aspect(params.get('aspect', 'equal'))
        plt.axis("off")
        if 'path' in params:
            plt.savefig(params['path'])
            plt.close()
        if params.get('show', True):
            plt.show()
    return axis


def to_gif(diagram, *diagrams, path=None, timestep=500, loop=False, **params):
    """
    Draws a sequence of diagrams.
    """
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
        return HTML('<img src="{}">'.format(path))


def pregroup_draw(words, cups, **params):
    """
    Draws pregroup words and cups.

    >>> from discopy import *
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Word('Alice', n), Word('Bob', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    >>> words, *cups = sentence.foliation().boxes
    >>> pregroup_draw(words, cups, to_tikz=True)
    \\node () at (1.1, -0.2) {n};
    \\draw (0.0, 0) -- (2.0, 0) -- (1.0, 1) -- (0.0, 0);
    \\node () at (1.0, 0.1) {Alice};
    \\node () at (3.1, -0.2) {n.r};
    \\node () at (3.6, -0.2) {s};
    \\node () at (4.1, -0.2) {n.l};
    \\draw (2.5, 0) -- (4.5, 0) -- (3.5, 1) -- (2.5, 0);
    \\node () at (3.5, 0.1) {loves};
    \\node () at (6.1, -0.2) {n};
    \\draw (5.0, 0) -- (7.0, 0) -- (6.0, 1) -- (5.0, 0);
    \\node () at (6.0, 0.1) {Bob};
    \\draw [out=-90, in=180] (1.0, 0) to (2.0, -1);
    \\draw [out=-90, in=0] (3.0, 0) to (2.0, -1);
    \\draw [out=-90, in=180] (4.0, 0) to (5.0, -1);
    \\draw [out=-90, in=0] (6.0, 0) to (5.0, -1);
    \\draw [out=-90, in=90] (3.5, 0) to (3.5, -2);
    \\node () at (3.6, -1.5) {s};
    <BLANKLINE>
    """
    textpad = params.get('textpad', (.1, .2))
    textpad_words = params.get('textpad_words', (0, .1))
    space = params.get('space', .5)
    width = params.get('width', 2.)
    fontsize = params.get('fontsize', 12)

    def draw_triangles(axis, words):
        scan = []
        for i, word in enumerate(words.boxes):
            for j, _ in enumerate(word.cod):
                x_wire = (space + width) * i\
                    + (width / (len(word.cod) + 1)) * (j + 1)
                scan.append(x_wire)
                if params.get('draw_types', True):
                    draw_text(axis, str(word.cod[j]),
                              x_wire + textpad[0], -textpad[1],
                              fontsize=params.get('fontsize_types', fontsize),
                              to_tikz=params.get('to_tikz', False))
            draw_polygon(
                axis, ((space + width) * i, 0),
                ((space + width) * i + width, 0),
                ((space + width) * i + width / 2, 1),
                color='none', to_tikz=params.get('to_tikz', False))
            draw_text(axis, str(word),
                      (space + width) * i + width / 2 + textpad_words[0],
                      textpad_words[1], ha='center', fontsize=fontsize,
                      to_tikz=params.get('to_tikz', False))
        return scan

    def draw_cups_and_wires(axis, cups, scan):
        for j, off in [(j, off)
                       for j, s in enumerate(cups) for off in s.offsets]:
            middle = (scan[off] + scan[off + 1]) / 2
            draw_wire(axis, (scan[off], 0), (middle, - j - 1),
                      bend_in=True, to_tikz=params.get('to_tikz', False))
            draw_wire(axis, (scan[off + 1], 0), (middle, - j - 1),
                      bend_in=True, to_tikz=params.get('to_tikz', False))
            scan = scan[:off] + scan[off + 2:]
        for i, _ in enumerate(cups[-1].cod):
            draw_wire(axis, (scan[i], 0), (scan[i], - len(cups) - 1),
                      to_tikz=params.get('to_tikz', False))
            if params.get('draw_types', True):
                draw_text(axis, str(cups[-1].cod[i]),
                          scan[i] + textpad[0], - len(cups) - space,
                          fontsize=params.get('fontsize_types', fontsize),
                          to_tikz=params.get('to_tikz', False))
    axis = [] if params.get('to_tikz', False)\
        else plt.subplots(figsize=params.get('figsize', None))[1]
    scan = draw_triangles(axis, words.normal_form())
    draw_cups_and_wires(axis, cups, scan)
    if params.get('to_tikz', False):
        if 'path' in params:
            save_tikz(axis, params['path'])
        else:
            print(''.join(axis))
    else:
        plt.margins(*params.get('margins', (.05, .05)))
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.axis('off')
        axis.set_xlim(0, (space + width) * len(words.boxes) - space)
        axis.set_ylim(- len(cups) - space, 1)
        axis.set_aspect(params.get('aspect', 'equal'))
        if 'path' in params.keys():
            plt.savefig(params['path'])
            plt.close()
        plt.show()


def equation(*diagrams, symbol="=", space=1, **params):
    """
    >>> from discopy import *
    >>> x = Ty('x')
    >>> diagrams = Id(x.r).transpose_l(), Id(x.l).transpose_r()
    >>> equation(*diagrams, to_tikz=True)
    \\node () at (0.1, 2.0) {x};
    \\node () at (1.1, 1.15) {x.r};
    \\node () at (2.1, 1.15) {x};
    \\draw [out=-90, in=90] (0, 2.0) to (0, 0.75);
    \\draw [out=180, in=90] (1.5, 1.5) to (1.0, 1.25);
    \\draw [out=0, in=90] (1.5, 1.5) to (2.0, 1.25);
    \\draw [out=-90, in=90] (1.0, 1.25) to (1.0, 0.75);
    \\draw [out=-90, in=90] (2.0, 1.25) to (2.0, 0.0);
    \\draw [out=-90, in=180] (0, 0.75) to (0.5, 0.5);
    \\draw [out=-90, in=0] (1.0, 0.75) to (0.5, 0.5);
    \\node () at (3.0, 1.0) {=};
    \\node () at (6.1, 2.0) {x};
    \\node () at (4.1, 1.15) {x};
    \\node () at (5.1, 1.15) {x.l};
    \\draw [out=-90, in=90] (6.0, 2.0) to (6.0, 0.75);
    \\draw [out=180, in=90] (4.5, 1.5) to (4.0, 1.25);
    \\draw [out=0, in=90] (4.5, 1.5) to (5.0, 1.25);
    \\draw [out=-90, in=90] (4.0, 1.25) to (4.0, 0.0);
    \\draw [out=-90, in=90] (5.0, 1.25) to (5.0, 0.75);
    \\draw [out=-90, in=180] (5.0, 0.75) to (5.5, 0.5);
    \\draw [out=-90, in=0] (6.0, 0.75) to (5.5, 0.5);
    <BLANKLINE>
    """
    axis, pad, max_height = None, 0, max(map(len, diagrams))
    path = params.get("path", None)
    if "path" in params:
        del params['path']
    for i, diagram in enumerate(diagrams):
        graph, positions, labels = diagram_to_nx(diagram)
        widths, height = {x for x, _ in positions.values()}, len(diagram) or 1
        min_width, max_width = min(widths), max(widths)
        positions = {n: (x - min_width, y) for n, (x, y) in positions.items()}
        axis = diagram.draw(axis=axis, data=(graph, positions, labels),
                            scale=(1, max_height / height), pad=(pad, 0),
                            show=False, **params)
        pad += max_width - min_width + space
        if i < len(diagrams) - 1:
            draw_text(axis, symbol, pad, max_height / 2,
                      to_tikz=params.get('to_tikz', False))
            pad += space
    if params.get('to_tikz', False):
        if path is not None:
            save_tikz(axis, path)
        else:  # pragma: no cover
            print(''.join(axis))
    else:
        if path is not None:
            plt.savefig(path)
            plt.close()
        plt.show()
