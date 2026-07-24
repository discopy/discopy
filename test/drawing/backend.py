from pytest import raises

from discopy.monoidal import Ty, Box


def test_draw_raises_on_relabeled_box(tmp_path):
    """ Drawing a relabeled box over an existing baseline raises. """
    x, y = Ty('x'), Ty('y')
    path = tmp_path / "box.svg"
    Box('f', x, y).draw(path=path, show=False)
    with raises(ValueError, match="Drawing differs"):
        Box('g', x, y).draw(path=path, show=False)
