from pytest import raises

from discopy.drawing.backend import Backend
from discopy.monoidal import Ty, Box


def test_draw_raises_on_relabeled_box(tmp_path):
    """ Drawing a relabeled box over an existing baseline raises. """
    x, y = Ty('x'), Ty('y')
    path = tmp_path / "box.svg"
    Box('f', x, y).draw(doctest=path, show=False)
    with raises(ValueError, match="Drawing differs"):
        Box('g', x, y).draw(doctest=path, show=False)


def test_arrowhead_segment():
    """ The arrowhead sits in the middle of the wire, pointing at target. """
    assert Backend.arrowhead_segment((0, 0), (0, 4), length=2)\
        == ((0, 1), (0, 3))
    assert Backend.arrowhead_segment((0, 4), (0, 0), length=2)\
        == ((0, 3), (0, 1))
    assert Backend.arrowhead_segment((1, 1), (1, 1)) == ((1, 1), (1, 1))


def tikz_source(diagram, tmp_path, **params):
    path = tmp_path / "diagram.tikz"
    diagram.draw(path=path, to_tikz=True, **params)
    return path.read_text()


def test_draw_feedback_arrow(tmp_path):
    """ A feedback loop is drawn with an arrow, a trace without. """
    drawing = Box('f', Ty('x') ** 2, Ty('x') ** 2).to_drawing()
    assert tikz_source(drawing.trace(), tmp_path).count("\\draw [->]") == 0
    assert tikz_source(
        drawing.trace(feedback=True), tmp_path).count("\\draw [->]") == 1
    assert tikz_source(
        drawing.trace(n=2, feedback=True), tmp_path).count("\\draw [->]") == 2


def test_draw_one_arrow_per_feedback_loop(tmp_path):
    """ Each feedback loop gets its own arrow, the traces get none. """
    x = Ty('x')
    f, g = (Box(name, x ** 2, x ** 2).to_drawing() for name in "fg")
    for drawing, expected in [
            (f.trace(feedback=True) @ g.trace(feedback=True), 2),
            (f.trace(feedback=True) >> g.trace(feedback=True), 2),
            (f.trace(feedback=True) @ g.trace(), 1),
            (f.trace() >> g.trace(feedback=True), 1),
            (f.trace(n=2, feedback=True).trace(), 2),
            (f.trace(left=True, feedback=True), 1)]:
        assert tikz_source(
            drawing, tmp_path).count("\\draw [->]") == expected


def test_feedback_to_drawing_is_directed(tmp_path):
    """ :class:`discopy.feedback.Feedback` draws its loop with an arrow. """
    from discopy.feedback import Ty as FTy, Box as FBox
    x = FTy('x')
    f = FBox('f', x @ x.d, x @ x).feedback()
    assert [box.draw_as_feedback
            for box in f.to_drawing().boxes if box.draw_as_wires]\
        == [True, True]
    nested = FBox('g', x @ x.d @ x.d, x @ x @ x).feedback().feedback()
    for diagram in (f >> f, f @ f, nested):
        assert tikz_source(
            diagram.to_drawing(), tmp_path).count("\\draw [->]") == 2
