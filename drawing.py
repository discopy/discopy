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
    graph, pos, labels = nx.Graph(), dict(), dict()

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
                pos[scan[off + i]][0], len(diagram) - depth)
            add_node(wire, position, str(box.dom[i]))
            graph.add_edge(scan[off + i], wire)
            graph.add_edge(wire, node)
        for i, _ in enumerate(box.cod):
            wire, position = 'wire_cod_{}_{}'.format(depth, i), (
                x_pos - len(box.cod[1:]) / 2 + i, len(diagram) - depth - 1)
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


def draw(diagram, **params):
    """
    Draws a diagram.
    """
    graph, positions, labels = diagram_to_nx(diagram)
    asymmetry = params.get('asymmetry',
                           .25 * any(box.is_dagger for box in diagram.boxes))

    def draw_box(box, depth, axis):
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
        height = len(diagram) - depth - .75
        left, right = left - .25, right + .25
        path = Path(
            [(left, height),
             (right + (asymmetry if box.is_dagger else 0), height),
             (right + (0 if box.is_dagger else asymmetry), height + .5),
             (left, height + .5), (left, height)],
            [Path.MOVETO] + 3 * [Path.LINETO] + [Path.CLOSEPOLY])
        axis.add_patch(PathPatch(
            path, facecolor=params.get('color', '#ffffff')))
        if params.get('draw_box_labels', True):
            axis.text(positions[node][0], positions[node][1], str(box.name),
                      ha='center', va='center',
                      fontsize=params.get('fontsize', 12))

    def draw_wires(axis):
        for case in ['input', 'output', 'wire_dom', 'wire_cod']:
            nodes = [n for n in graph.nodes if n[:len(case)] == case]
            nx.draw_networkx_nodes(
                graph, positions, nodelist=nodes, node_size=0, ax=axis)
            for node in nodes:
                i, j = positions[node]
                if params.get('draw_types', True)\
                        and case in ['input', 'wire_cod']:
                    if node in labels.keys():
                        axis.text(
                            i + params.get('textpad', .1),
                            j - (params.get('textpad', .1)
                                 if case == 'input' else 0),
                            labels[node],
                            fontsize=params.get(
                                'fontsize_types',
                                params.get('fontsize', 12)))
                if not params.get('draw_as_nodes', False):
                    if case == 'wire_dom':
                        positions[node] = (i, j - .25)
                    elif case == 'wire_cod':
                        positions[node] = (i, j + .25)
        for node0, node1 in graph.edges():
            source, target = positions[node0], positions[node1]
            path = Path([source, (target[0], source[1]), target],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            axis.add_patch(PathPatch(path, facecolor='none'))
    _, axis = plt.subplots(figsize=params.get('figsize', None))
    draw_wires(axis)
    if params.get('draw_as_nodes', False):
        boxes = [node for node in graph.nodes if node[:3] == 'box']
        nx.draw_networkx_nodes(
            graph, positions, nodelist=boxes,
            node_color=params.get('color', '#ff0000'), ax=axis)
        nx.draw_networkx_labels(
            graph, positions,
            {n: l for n, l in labels.items() if n in boxes})
    else:
        for depth, box in enumerate(diagram.boxes):
            draw_box(box, depth, axis)
    plt.margins(*params.get('margins', (.05, .05)))
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    axis.set_aspect(params.get('aspect', 'equal'))
    plt.axis("off")
    if 'path' in params:
        plt.savefig(params['path'])
        plt.close()
    plt.show()


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
    """
    textpad = params.get('textpad', .1)
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
                    axis.text(x_wire + textpad, -2 * textpad, str(word.cod[j]),
                              fontsize=params.get('fontsize_types', fontsize))
            path = Path(
                [((space + width) * i, 0),
                 ((space + width) * i + width, 0),
                 ((space + width) * i + width / 2, 1),
                 ((space + width) * i, 0)],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
            axis.add_patch(PathPatch(path, facecolor='none'))
            axis.text((space + width) * i + width / 2, textpad,
                      str(word), ha='center', fontsize=fontsize)
        return scan

    def draw_cups_and_wires(axis, cups, scan):
        for j, off in [(j, off)
                       for j, s in enumerate(cups) for off in s.offsets]:
            middle = (scan[off] + scan[off + 1]) / 2
            verts = [(scan[off], 0),
                     (scan[off], - j - 1),
                     (middle, - j - 1)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            axis.add_patch(PathPatch(Path(verts, codes), facecolor='none'))
            verts = [(middle, - j - 1),
                     (scan[off + 1], - j - 1),
                     (scan[off + 1], 0)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            axis.add_patch(PathPatch(Path(verts, codes), facecolor='none'))
            scan = scan[:off] + scan[off + 2:]
        for i, _ in enumerate(cups[-1].cod):
            verts = [(scan[i], 0), (scan[i], - len(cups) - 1)]
            codes = [Path.MOVETO, Path.LINETO]
            axis.add_patch(PathPatch(Path(verts, codes)))
            if params.get('draw_types', True):
                axis.text(
                    scan[i] + textpad, - len(cups) - space,
                    str(cups[-1].cod[i]),
                    fontsize=params.get('fontsize_types', fontsize))
    _, axis = plt.subplots(figsize=params.get('figsize', None))
    scan = draw_triangles(axis, words.normal_form())
    draw_cups_and_wires(axis, cups, scan)
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
