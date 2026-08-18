from pytest import raises

from discopy.monoidal import Ty, Box


def test_draw_raises_on_relabeled_box(tmp_path):
    """ Drawing a relabeled box over an existing baseline raises. """
    x, y = Ty('x'), Ty('y')
    path = tmp_path / "box.svg"
    Box('f', x, y).draw(doctest=path, show=False)
    with raises(ValueError, match="Drawing differs"):
        Box('g', x, y).draw(doctest=path, show=False)


def tikz_source(diagram, tmp_path, **params):
    path = tmp_path / "diagram.tikz"
    diagram.draw(path=path, to_tikz=True, wire_labels=False, **params)
    return path.read_text()


def delay_labels(diagram, tmp_path, **params):
    """ The delay box labels in a TikZ drawing, in drawing order. """
    return [label for label in (
        line.split("{")[1].split("}")[0]
        for line in tikz_source(diagram, tmp_path, **params).splitlines()
        if line.startswith("\\node [style=none, fill=white]"))
        if label.isdigit()]


def test_draw_feedback_box(tmp_path):
    """ A feedback loop is drawn with a delay box, a trace without. """
    drawing = Box('f', Ty('x') ** 2, Ty('x') ** 2).to_drawing()
    assert delay_labels(drawing.trace(), tmp_path) == []
    assert delay_labels(drawing.trace(feedback=True), tmp_path) == ["1"]
    assert delay_labels(drawing.trace(feedback=2), tmp_path) == ["2"]
    assert delay_labels(
        drawing.trace(n=2, feedback=True), tmp_path) == ["1", "1"]


def test_draw_one_delay_box_per_feedback_loop(tmp_path):
    """ Each feedback loop gets its own delay box, the traces get none. """
    x = Ty('x')
    f, g = (Box(name, x ** 2, x ** 2).to_drawing() for name in "fg")
    for drawing, expected in [
            (f.trace(feedback=True) @ g.trace(feedback=True), 2),
            (f.trace(feedback=True) >> g.trace(feedback=True), 2),
            (f.trace(feedback=True) @ g.trace(), 1),
            (f.trace() >> g.trace(feedback=True), 1),
            (f.trace(n=2, feedback=True).trace(), 2),
            (f.trace(left=True, feedback=True), 1)]:
        assert len(delay_labels(drawing, tmp_path)) == expected


def test_feedback_to_drawing_delay():
    """ A feedback loop marks its cup and cap with its delay. """
    from discopy.feedback import Ty as FTy, Box as FBox
    x = FTy('x')
    f = FBox('f', x @ x.d, x @ x).feedback()
    assert [box.draw_as_feedback
            for box in f.to_drawing().boxes if box.draw_as_wires] == [1, 1]
    two_steps = FBox('g', x @ x.d.d, x @ x).to_drawing().trace(feedback=2)
    assert [box.draw_as_feedback
            for box in two_steps.boxes if box.draw_as_wires] == [2, 2]


def test_draw_nested_feedback(tmp_path):
    """ Feedback loops in sequence, parallel and nested each get a box. """
    from discopy.feedback import Ty as FTy, Box as FBox
    x = FTy('x')
    f = FBox('f', x @ x.d, x @ x).feedback()
    nested = FBox('g', x @ x.d @ x.d, x @ x @ x).feedback().feedback()
    for diagram in (f >> f, f @ f, nested):
        assert delay_labels(diagram.to_drawing(), tmp_path) == ["1", "1"]
