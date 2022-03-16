# -*- coding: utf-8 -*-
""" Drawing module. """

from discopy.drawing import Node, draw_box, add_drawing_attributes


def draw_discard(backend, positions, node, **params):
    """ Draws a :class:`discopy.quantum.circuit.Discard` box. """
    box, depth = node.box, node.depth
    for i in range(box.n_qubits):
        obj = box.dom[i]
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
    distance = box.distance
    c_size = len(box.controlled.dom)

    index = (0, distance) if distance > 0 else (c_size - distance - 1, 0)
    dom = Node("dom", obj=box.dom[0], i=index[0], depth=depth)
    cod = Node("cod", obj=box.cod[0], i=index[0], depth=depth)
    middle = positions[dom][0], (positions[dom][1] + positions[cod][1]) / 2
    controlled_box = add_drawing_attributes(box.controlled.downgrade())
    controlled = Node("box", box=controlled_box, depth=depth)
    # TODO select obj properly for classical gates
    c_dom = Node("dom", obj=box.dom[0], i=index[1], depth=depth)
    c_cod = Node("cod", obj=box.cod[0], i=index[1], depth=depth)
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
            dom_node = Node("dom", obj=box.dom[i], i=i, depth=depth)
            x, y = positions[c_dom][0] + i, positions[c_dom][1]
            fake_positions[dom_node] = x, y

            cod_node = Node("cod", obj=box.cod[i], i=i, depth=depth)
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
        node1 = Node("dom", obj=box.dom[i], i=i, depth=depth)
        node2 = Node("cod", obj=box.cod[i], i=i, depth=depth)
        backend.draw_wire(positions[node1], positions[node2])

    # TODO change bend_in and bend_out for tikz backend
    backend.draw_wire(middle, target_boundary, bend_in=True, bend_out=True)

    backend.draw_node(
        *middle, color="black", shape="circle",
        nodesize=params.get("nodesize", 1))
    return backend
