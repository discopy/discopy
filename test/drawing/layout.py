from math import pi

from discopy import (
    monoidal, braided, balanced, symmetric, traced, rigid, pivotal,
    ribbon, compact, frobenius, markov, closed, feedback, tensor)
from discopy.drawing import gui
from discopy.drawing.layout import (
    HIERARCHY, ForceLayout, box_kind, doctrine, port_angles, to_layout)


def test_doctrine_of_each_module():
    expected = {
        monoidal: "monoidal", braided: "braided", balanced: "balanced",
        symmetric: "symmetric", traced: "traced", rigid: "rigid",
        pivotal: "pivotal", ribbon: "ribbon", compact: "compact",
        frobenius: "frobenius", markov: "symmetric", closed: "symmetric",
        feedback: "symmetric", tensor: "frobenius"}
    for module, name in expected.items():
        assert doctrine(module.Diagram).name == name


def test_doctrine_is_monotone():
    features = dict(HIERARCHY)
    for module, bigger in [
            (monoidal, braided), (braided, balanced), (balanced, symmetric),
            (rigid, pivotal), (pivotal, ribbon), (ribbon, compact),
            (compact, frobenius)]:
        small = doctrine(module.Diagram).features
        big = doctrine(bigger.Diagram).features
        assert small < big
        assert small == features[doctrine(module.Diagram).name]


def test_doctrine_level():
    assert doctrine(monoidal.Diagram).level == 0
    assert doctrine(frobenius.Diagram).level == len(HIERARCHY) - 1


def test_box_kind():
    x, y = frobenius.Ty('x'), frobenius.Ty('y')
    assert [box_kind(b) for b in frobenius.Box(
        'f', x, y).bubble().to_drawing().boxes] == 3 * ["box"]
    assert [box_kind(b) for b in frobenius.Diagram.cups(
        x, x.r).to_drawing().boxes] == ["cup"]
    assert [box_kind(b) for b in frobenius.Diagram.caps(
        x, x.r).to_drawing().boxes] == ["cap"]
    assert [box_kind(b) for b in frobenius.Swap(
        x, y).to_drawing().boxes] == ["swap"]
    assert [box_kind(b) for b in braided.Braid(
        braided.Ty('x'), braided.Ty('y')).to_drawing().boxes] == ["braid"]
    assert [box_kind(b) for b in frobenius.Spider(
        1, 2, x).to_drawing().boxes] == ["spider"]
    assert [box_kind(b) for b in frobenius.Box(
        'f', x, y).to_drawing().boxes] == ["box"]


def test_port_angles_are_cyclic():
    dom, cod = port_angles(2, 3)
    assert all(0 < a < pi for a in dom)
    assert all(pi < a < 2 * pi for a in cod)
    assert dom == sorted(dom, reverse=True)
    assert cod == sorted(cod)


def test_to_layout_respects_composition():
    x, y, z = map(monoidal.Ty, "xyz")
    f, g = monoidal.Box('f', x, y @ z), monoidal.Box('g', y @ z, x)
    top, bottom, both = map(to_layout, (f, g, f >> g))
    assert [b["label"] for b in both["boxes"]] \
        == [b["label"] for b in top["boxes"] + bottom["boxes"]]
    assert len(both["wires"]) \
        == len(top["wires"]) + len(bottom["wires"]) - len(f.cod)


def test_to_layout_respects_tensor():
    x, y = map(monoidal.Ty, "xy")
    f, g = monoidal.Box('f', x, y), monoidal.Box('g', y, x)
    left, right, both = map(to_layout, (f, g, f @ g))
    assert [b["label"] for b in both["boxes"]] \
        == [b["label"] for b in left["boxes"] + right["boxes"]]
    assert len(both["wires"]) == len(left["wires"]) + len(right["wires"])


def test_to_layout_endpoints_are_valid():
    d = frobenius.Spider(1, 2, frobenius.Ty('x'))\
        >> frobenius.Box('f', frobenius.Ty('x'), frobenius.Ty('x'))\
        @ frobenius.Ty('x')
    spec = to_layout(d)
    for wire in spec["wires"]:
        for end in (wire["source"], wire["target"]):
            if end["kind"] == "box":
                ports = spec["boxes"][end["j"]][end["side"]]
                assert 0 <= end["i"] < len(ports)
            else:
                assert 0 <= end["i"] < len(spec[end["kind"]])


def test_force_layout_relaxes():
    d = monoidal.Box('f', monoidal.Ty('x'), monoidal.Ty('y'))\
        >> monoidal.Box('g', monoidal.Ty('y'), monoidal.Ty('z'))
    engine = ForceLayout(to_layout(d))
    engine.position += [[2., -1.], [-2., 1.]]  # Perturb the initial layout.
    before = engine.energy()
    assert engine.run(200).energy() < before


def test_force_layout_pivots_rotate():
    snake = pivotal.Box(
        'f', pivotal.Ty('x'), pivotal.Ty('y')).transpose()
    frozen = ForceLayout(to_layout(snake), features={"bends"}).run(50)
    assert not frozen.angle.any()
    pivots = ForceLayout(to_layout(snake)).run(50)
    assert pivots.angle.any()


def test_gui_to_html():
    diagrams, titles, blurbs = gui.gallery()
    html = gui.to_html(*diagrams, titles=titles, blurbs=blurbs)
    assert "<canvas" in html and "__GALLERY_JSON__" not in html
    for title in titles:
        assert title in html
    for name, _ in HIERARCHY:
        assert name in html


def test_gui_draw(tmp_path):
    path = gui.draw(
        symmetric.Swap(symmetric.Ty('x'), symmetric.Ty('y')),
        path=str(tmp_path / "gui.html"))
    with open(path, encoding="utf-8") as file:
        assert "swap" in file.read()


def test_gui_draw_tmp_file():
    import os
    path = gui.draw(monoidal.Box('f', monoidal.Ty('x'), monoidal.Ty('y')))
    assert os.path.exists(path)
    os.remove(path)
