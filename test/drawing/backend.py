from pytest import mark, raises

from discopy.monoidal import Ty, Box


def test_draw_raises_on_relabeled_box(tmp_path):
    """ Drawing a relabeled box over an existing baseline raises. """
    x, y = Ty('x'), Ty('y')
    path = tmp_path / "box.svg"
    Box('f', x, y).draw(doctest=path, show=False)
    with raises(ValueError, match="Drawing differs"):
        Box('g', x, y).draw(doctest=path, show=False)


@mark.parametrize("n", [1, 2, 3, 5])
def test_draw_discard_on_n_wires(tmp_path, n):
    """ Discards draw on any number of wires, see discopy/discopy#513. """
    x = Ty('x')
    discard = Box('discard', x ** n, Ty(), draw_as_discards=True)
    discard.draw(path=tmp_path / f"discard-{n}.svg", show=False)
