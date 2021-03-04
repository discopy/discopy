# -*- coding: utf-8 -*-
""" Drawing module. """

from discopy.drawing import Node, draw_box, add_drawing_attributes


def draw_discard(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.circuit.Discard` box. """
    box, depth = node.box, node.depth
    left_dom, right_dom = (
        Node("dom", obj=box.dom[i], i=i, depth=depth)
        for i in [0, len(box.dom) - 1])
    left, right = (positions[n][0] for n in [left_dom, right_dom])
    left, right = left - .25, right + .25
    height = positions[node][1] + .25
    for i in range(3):
        source = (left + .1 * i, height - .1 * i)
        target = (right - .1 * i, height - .1 * i)
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
        obj = box.dom[i] if is_bra else box.cod[i]
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
    dom = Node("dom", obj=box.dom[0], i=0, depth=depth)
    cod = Node("cod", obj=box.cod[0], i=0, depth=depth)
    middle = positions[dom][0], (positions[dom][1] + positions[cod][1]) / 2
    controlled_box = add_drawing_attributes(box.controlled.downgrade())
    controlled = Node("box", box=controlled_box, depth=depth)
    c_dom = Node("dom", obj=box.dom[0], i=1, depth=depth)
    c_cod = Node("cod", obj=box.cod[0], i=1, depth=depth)
    c_middle =\
        positions[c_dom][0], (positions[c_dom][1] + positions[c_cod][1]) / 2
    target = (positions[c_dom][0],
              (positions[c_dom][1] + positions[c_cod][1]) / 2)
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
        left_of_target = target
    else:
        left_of_target = c_middle[0] - .25, c_middle[1]
        fake_positions = {
            controlled: target, dom: positions[c_dom], cod: positions[c_cod]}
        backend = draw_box(backend, fake_positions, controlled, **params)
    backend.draw_wire(positions[dom], positions[cod])
    # TODO change bend_in and bend_out for tikz backend
    backend.draw_wire(middle, left_of_target, bend_in=True, bend_out=True)
    backend.draw_node(
        *middle, color="black", shape="circle",
        nodesize=params.get("nodesize", 1))
    return backend
