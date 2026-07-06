# -*- coding: utf-8 -*-

"""
Functorial force-directed layout for the hierarchy of diagram doctrines.

Each level of the hierarchy of :cite:t:`Selinger10` is interpreted as a set
of geometric degrees of freedom, called :class:`features <Doctrine>`, that a
force-directed layout is allowed to use:

* ``monoidal`` diagrams are progressive plane graphs: wires flow downward
  and may not cross, boxes may not rotate,
* ``braids`` lets wires cross over and under each other,
* ``twists`` lets wires carry framing, drawn as ribbons,
* ``swaps`` lets wires cross freely, forgetting over and under,
* ``feedback`` lets wires flow back up around a traced box,
* ``bends`` lets wires bend by half-turns, i.e. cups and caps,
* ``pivots`` makes each box a pivot: its ports are fixed on a circle
  which can rotate freely around the box, so that the transpose of a box
  is its rotation by a half-turn,
* ``spiders`` forgets the order of ports altogether: wires can attach
  to a spider at any angle.

The layout of a diagram is derived from :meth:`Diagram.to_drawing`, which is
itself a :class:`Functor` into the category :class:`Drawing` of labeled plane
graphs. The force system is generated locally, box by box and wire by wire,
so that the assignment ``diagram -> force system`` preserves composition and
tensor: it is the composite of the drawing functor with a translation that
acts generator by generator.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Doctrine
    ForceLayout

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    doctrine
    to_layout

Example
-------
>>> from discopy.rigid import Ty, Box
>>> f = Box('f', Ty('x'), Ty('y'))
>>> doctrine(f.transpose())
Doctrine('rigid')
>>> spec = to_layout(f.transpose())
>>> [box["kind"] for box in spec["boxes"]]
['cap', 'box', 'cup']
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, cos, sin

from discopy.utils import Node


#: The new geometric feature introduced by each module of the hierarchy.
NEW_FEATURES = {
    "braided": frozenset({"braids"}),
    "balanced": frozenset({"twists"}),
    "symmetric": frozenset({"swaps"}),
    "traced": frozenset({"feedback"}),
    "feedback": frozenset({"feedback"}),
    "rigid": frozenset({"bends"}),
    "pivotal": frozenset({"pivots"}),
    "frobenius": frozenset({"spiders"}),
    "hypergraph": frozenset({"spiders"}),
}

#: The canonical named levels of the hierarchy with their features,
#: mirroring the inheritance of the corresponding :class:`Diagram` classes,
#: e.g. balanced diagrams are traced so ``balanced`` includes ``feedback``.
HIERARCHY = (
    ("monoidal", frozenset()),
    ("braided", frozenset({"braids"})),
    ("traced", frozenset({"feedback"})),
    ("balanced", frozenset({"braids", "twists", "feedback"})),
    ("symmetric", frozenset({"braids", "twists", "feedback", "swaps"})),
    ("rigid", frozenset({"bends"})),
    ("pivotal", frozenset({"bends", "feedback", "pivots"})),
    ("ribbon", frozenset(
        {"bends", "feedback", "pivots", "braids", "twists"})),
    ("compact", frozenset(
        {"bends", "feedback", "pivots", "braids", "twists", "swaps"})),
    ("frobenius", frozenset({
        "bends", "feedback", "pivots",
        "braids", "twists", "swaps", "spiders"})),
)


@dataclass(frozen=True)
class Doctrine:
    """
    A named level of the hierarchy together with its geometric features.

    Parameters:
        name : The name of the smallest level with the given features.
        features : The set of degrees of freedom the layout may use.

    Example
    -------
    >>> from discopy import monoidal, symmetric, pivotal, frobenius
    >>> for module in (monoidal, symmetric, pivotal, frobenius):
    ...     print(doctrine(module.Diagram))
    monoidal
    symmetric
    pivotal
    frobenius
    >>> sorted(doctrine(pivotal.Diagram).features)
    ['bends', 'feedback', 'pivots']
    """
    name: str
    features: frozenset

    def __repr__(self):
        return f"Doctrine({repr(self.name)})"

    def __str__(self):
        return self.name

    @property
    def level(self) -> int:
        """ The index of the doctrine inside :obj:`HIERARCHY`. """
        return [name for name, _ in HIERARCHY].index(self.name)


def doctrine(diagram_or_type) -> Doctrine:
    """
    The doctrine of a diagram, computed by walking its method resolution
    order and taking the union of the features of each module.

    Parameters:
        diagram_or_type : A :class:`monoidal.Diagram` or a subclass.

    Example
    -------
    >>> from discopy import braided, traced, compact
    >>> doctrine(braided.Diagram), doctrine(traced.Diagram)
    (Doctrine('braided'), Doctrine('traced'))
    >>> doctrine(compact.Id(compact.Ty('x')))
    Doctrine('compact')

    The hierarchy is a poset rather than a chain: the doctrine of a diagram
    is the smallest named level that contains all of its features, e.g.
    feedback diagrams are laid out with the features of symmetric ones.

    >>> from discopy import feedback
    >>> doctrine(feedback.Diagram).name
    'symmetric'
    """
    cls = diagram_or_type if isinstance(diagram_or_type, type)\
        else type(diagram_or_type)
    features = frozenset().union(*(
        NEW_FEATURES.get(c.__module__.split('.')[-1], frozenset())
        for c in cls.__mro__))
    for name, level_features in sorted(
            HIERARCHY, key=lambda level: len(level[1])):
        if features <= level_features:
            return Doctrine(name, level_features)
    return Doctrine(*HIERARCHY[-1])  # pragma: no cover


def box_kind(box) -> str:
    """
    Classify a drawing box by its attributes, one of
    ``box``, ``cup``, ``cap``, ``swap``, ``braid`` or ``spider``.

    Parameters:
        box : A box with the attributes of :obj:`config.DRAWING_ATTRIBUTES`.

    Example
    -------
    >>> from discopy.compact import Ty, Cup, Swap
    >>> x, y = Ty('x'), Ty('y')
    >>> [box_kind(b) for b in Cup(x, x.r).to_drawing().boxes]
    ['cup']
    >>> [box_kind(b) for b in Swap(x, y).to_drawing().boxes]
    ['swap']
    """
    if getattr(box, "draw_as_spider", False):
        return "spider"
    if getattr(box, "draw_as_braid", False):
        return "braid"
    if getattr(box, "bubble_opening", False)\
            or getattr(box, "bubble_closing", False):
        return "box"
    if getattr(box, "draw_as_wires", False):
        if (len(box.dom), len(box.cod)) == (2, 0):
            return "cup"
        if (len(box.dom), len(box.cod)) == (0, 2):
            return "cap"
        if (len(box.dom), len(box.cod)) == (2, 2):
            return "swap"
    return "box"


def port_angles(n_dom: int, n_cod: int) -> tuple[list, list]:
    """
    The angles of the ports of a box on the circle around its pivot,
    in radians, measured anticlockwise with the y axis pointing up.

    Domain ports go left to right along the top half of the circle,
    codomain ports left to right along the bottom half, so that rotating
    a box with one input and one output by a half-turn is its transpose.

    Example
    -------
    >>> from math import pi
    >>> dom, cod = port_angles(1, 1)
    >>> assert dom == [pi / 2] and cod == [3 * pi / 2]
    >>> dom, _ = port_angles(3, 0)
    >>> [round(a / pi, 2) for a in dom]
    [0.75, 0.5, 0.25]
    """
    dom = [pi - (i + 1) * pi / (n_dom + 1) for i in range(n_dom)]
    cod = [pi + (i + 1) * pi / (n_cod + 1) for i in range(n_cod)]
    return dom, cod


def _endpoint(node: Node) -> dict:
    """ Encode one end of a wire, a boundary port or the port of a box. """
    if node.kind in ("dom", "cod"):
        return {"kind": node.kind, "i": node.i}
    side = "dom" if node.kind == "box_dom" else "cod"
    return {"kind": "box", "j": node.j, "side": side, "i": node.i}


def to_layout(diagram) -> dict:
    """
    The layout specification of a diagram: its doctrine, its boxes with
    their kinds and ports, its wires and its initial plane-graph positions.

    The specification is derived from :meth:`Diagram.to_drawing`, i.e. from
    the image of the diagram under the drawing functor, so that the
    specification of a composition is the gluing of the specifications.

    Parameters:
        diagram : A :class:`monoidal.Diagram` or a subclass.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> x, y, z = map(Ty, "xyz")
    >>> f, g = Box('f', x, y @ z), Box('g', y, x)
    >>> spec = to_layout(f >> g @ z)
    >>> spec["doctrine"], spec["features"]
    ('monoidal', [])
    >>> [box["label"] for box in spec["boxes"]]
    ['f', 'g']
    >>> len(spec["wires"])
    4
    """
    drawing = diagram.to_drawing()
    doct = doctrine(diagram)
    boxes = []
    for j, box in enumerate(drawing.boxes):
        position = drawing.positions[Node("box", box=box, j=j)]
        dom_angles, cod_angles = port_angles(len(box.dom), len(box.cod))
        boxes.append({
            "kind": box_kind(box),
            "label": str(getattr(box, "drawing_name", box.name)),
            "x": position.x, "y": position.y,
            "dom": [{"label": str(x), "angle": a}
                    for x, a in zip(box.dom, dom_angles)],
            "cod": [{"label": str(x), "angle": a}
                    for x, a in zip(box.cod, cod_angles)],
            "color": getattr(box, "color", "white"),
            "shape": getattr(box, "shape", None),
            "is_dagger": bool(getattr(box, "is_dagger", False)),
            "is_conjugate": bool(getattr(box, "is_conjugate", False))})
    wires = [
        {"source": _endpoint(u), "target": _endpoint(v),
         "label": str(getattr(u, "x", ""))}
        for u, v in drawing.edges
        if u.kind in ("dom", "box_cod") and v.kind in ("cod", "box_dom")]
    return {
        "doctrine": doct.name,
        "features": sorted(doct.features),
        "width": drawing.width,
        "height": drawing.height,
        "dom": [{"label": str(n.x), "x": drawing.positions[n].x}
                for n in drawing.dom_nodes],
        "cod": [{"label": str(n.x), "x": drawing.positions[n].x}
                for n in drawing.cod_nodes],
        "boxes": boxes,
        "wires": wires}


#: Box kinds whose wires are exempt from the progressive force.
BENDY_KINDS = ("cup", "cap", "spider")


class ForceLayout:
    """
    A force-directed layout engine for a layout specification, mirroring
    the simulation that runs in :mod:`discopy.drawing.gui`.

    The forces are generated locally: a spring along each wire, a repulsion
    between each pair of boxes and, depending on the features of the
    doctrine, a progressive force keeping wires flowing downward and a
    torque rotating each pivot with its ports.

    Parameters:
        spec : A layout specification, as returned by :func:`to_layout`.
        features : Optional features overriding those of the specification,
            i.e. laying out the image of the diagram under the inclusion
            functor into a higher level of the hierarchy.

    Example
    -------
    >>> from discopy.rigid import Ty, Box
    >>> snake = Box('f', Ty('x'), Ty('y')).transpose()
    >>> rigid = ForceLayout(to_layout(snake)).run(100)
    >>> pivotal = ForceLayout(
    ...     to_layout(snake), features={"bends", "pivots"}).run(100)
    >>> assert pivotal.energy() < rigid.energy()

    At the pivotal level the boxes may rotate, so the layout can relax
    further than at the rigid level where every angle is frozen.
    """
    def __init__(self, spec: dict, features=None,
                 stiffness=2., repulsion=.5, gravity=1., rest_length=.75):
        import numpy
        self._np, self.spec = numpy, spec
        self.features = frozenset(
            spec["features"] if features is None else features)
        self.stiffness, self.repulsion = stiffness, repulsion
        self.gravity, self.rest_length = gravity, rest_length
        n = len(spec["boxes"])
        self.position = numpy.array(
            [[box["x"], box["y"]] for box in spec["boxes"]]
        ).reshape(n, 2).astype(float)
        self.angle = numpy.zeros(n)
        self.velocity = numpy.zeros((n, 2))
        self.spin = numpy.zeros(n)

    def radius(self, j: int) -> float:
        """ The radius of the port circle of a given box. """
        box = self.spec["boxes"][j]
        return max(.5, .25 * max(len(box["dom"]), len(box["cod"])))

    def port(self, endpoint: dict):
        """ The position of one end of a wire, and whether it may bend. """
        numpy = self._np
        if endpoint["kind"] == "dom":
            x = self.spec["dom"][endpoint["i"]]["x"]
            return numpy.array([x, self.spec["height"]]), None
        if endpoint["kind"] == "cod":
            x = self.spec["cod"][endpoint["i"]]["x"]
            return numpy.array([x, 0.]), None
        j, box = endpoint["j"], self.spec["boxes"][endpoint["j"]]
        if box["kind"] in BENDY_KINDS or "pivots" not in self.features:
            offset = numpy.zeros(2)
            if box["kind"] not in BENDY_KINDS:
                ports = box[endpoint["side"]]
                i, n = endpoint["i"], len(ports)
                offset = numpy.array([
                    .5 * (i - (n - 1) / 2),
                    .25 if endpoint["side"] == "dom" else -.25])
        else:
            angle = box[endpoint["side"]][endpoint["i"]]["angle"]\
                + self.angle[j]
            offset = self.radius(j) * numpy.array([cos(angle), sin(angle)])
        return self.position[j] + offset, j

    def forces(self):
        """ The force on each box, the torque and the potential energy. """
        numpy = self._np
        n = len(self.spec["boxes"])
        force, torque, energy = numpy.zeros((n, 2)), numpy.zeros(n), 0.
        for wire in self.spec["wires"]:
            (source, j_source), (target, j_target) = map(
                self.port, (wire["source"], wire["target"]))
            vector = target - source
            distance = max(float(numpy.linalg.norm(vector)), 1e-6)
            stretch = distance - self.rest_length
            energy += .5 * self.stiffness * stretch ** 2
            pull = self.stiffness * stretch * vector / distance
            bendy = any(
                endpoint["kind"] == "box" and self.spec["boxes"][
                    endpoint["j"]]["kind"] in BENDY_KINDS
                for endpoint in (wire["source"], wire["target"]))
            drop = source[1] - target[1]  # Positive when flowing downward.
            slack = 0. if bendy else max(0., self.rest_length - drop)
            energy += .5 * self.gravity * slack ** 2
            for j, point, sign in (
                    (j_source, source, 1), (j_target, target, -1)):
                if j is None:
                    continue
                push = sign * pull + self.gravity * slack * numpy.array(
                    [0., -sign])
                force[j] += push
                if "pivots" in self.features:
                    arm = point - self.position[j]
                    torque[j] += arm[0] * push[1] - arm[1] * push[0]
        for j in range(n):
            for k in range(j + 1, n):
                vector = self.position[k] - self.position[j]
                distance = max(float(numpy.linalg.norm(vector)), 1e-2)
                push = self.repulsion * vector / distance ** 3
                force[j] -= push
                force[k] += push
                energy += self.repulsion / distance
        return force, torque, energy

    def step(self, dt=.02, damping=.85):
        """ One step of semi-implicit Euler integration. """
        force, torque, _ = self.forces()
        self.velocity = damping * (self.velocity + dt * force)
        self.spin = damping * (self.spin + dt * torque)
        self.position = self.position + dt * self.velocity
        if "pivots" in self.features:
            self.angle = self.angle + dt * self.spin
        return self

    def run(self, steps=100, dt=.02, damping=.85):
        """ Run the simulation for a given number of steps. """
        for _ in range(steps):
            self.step(dt, damping)
        return self

    def energy(self) -> float:
        """ The potential energy of the current state. """
        return self.forces()[2]
