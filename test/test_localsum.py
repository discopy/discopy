# -*- coding: utf-8 -*-

from pytest import raises, fixture
import discopy
import discopy.cat as cat
import discopy.monoidal as monoidal
from discopy.localsum_rewriting import distribute_composition_cat, distribute_composition_monoidal, distribute_tensor, _check_nothing_in_way


@fixture
def localsum_testdata_cat():
    x, y, z = cat.Ob('x'),cat.Ob('y'),cat.Ob('z')
    f = cat.Box('f', x, y)
    g = cat.Box('g', x, y)
    h = cat.Box('h', y, z)
    fg = cat.LocalSum([f, g])
    k = fg >> h
    l = cat.Box("h'", z, x) >> fg
    return x, y, z, f, g, h, fg, k, l


def test_localsum_creation(localsum_testdata_cat):
    _, _, _, _, _, _, _, k, l = localsum_testdata_cat
    assert isinstance(k.boxes[0], cat.LocalSum)
    assert isinstance(l.boxes[1], cat.LocalSum)


def test_distribute_composition_cat_left(localsum_testdata_cat):
    _, _, _, _, _, _, _, _, l = localsum_testdata_cat
    a = distribute_composition_cat(l, 1, 0)
    assert len(a.boxes) == 1
    assert len(a.boxes[0].terms) == 2
    assert len(a.boxes[0].terms[0].boxes) == 2
    assert a.boxes[0].terms[0].boxes[0].name == "h'"
    assert a.boxes[0].terms[0].boxes[1].name == "f"
    assert len(a.boxes[0].terms[1].boxes) == 2
    assert a.boxes[0].terms[1].boxes[0].name == "h'"
    assert a.boxes[0].terms[1].boxes[1].name == "g"


def test_distribute_composition_cat_right(localsum_testdata_cat):
    _, _, _, _, _, _, _, k, _ = localsum_testdata_cat
    a = distribute_composition_cat(k, 0, 1)
    assert len(a.boxes) == 1
    assert len(a.boxes[0].terms) == 2
    assert len(a.boxes[0].terms[0].boxes) == 2
    assert a.boxes[0].terms[0].boxes[0].name == "f"
    assert a.boxes[0].terms[0].boxes[1].name == "h"
    assert len(a.boxes[0].terms[1].boxes) == 2
    assert a.boxes[0].terms[1].boxes[0].name == "g"
    assert a.boxes[0].terms[1].boxes[1].name == "h"


def test_distribute_composition_cat_right2(localsum_testdata_cat):
    x, y, z, _, _, _, _, _, _ = localsum_testdata_cat
    diag = cat.Box("a", x, y) >> cat.Box("b", y, x) >> cat.LocalSum([cat.Box("c", x, z), cat.Box(
        "d", x, z)]) >> cat.Box("e", z, y) >> cat.Box("f", y, y) >> cat.Box("g", y, x)
    a = distribute_composition_cat(diag, 2, 1)
    assert len(a.boxes) == 5
    assert isinstance(a.boxes[1], cat.LocalSum)
    assert a.boxes[1].terms[0].boxes[0].name == "b"
    a = distribute_composition_cat(diag, 2, 4)
    assert len(a.boxes) == 4
    assert isinstance(a.boxes[2], cat.LocalSum)
    assert len(a.boxes[2].terms[0].boxes) == 3
    assert a.boxes[2].terms[0].boxes[0].name == "c"
    assert a.boxes[2].terms[0].boxes[1].name == "e"
    assert a.boxes[2].terms[0].boxes[2].name == "f"
    assert a.boxes[2].terms[1].boxes[0].name == "d"
    assert a.boxes[2].terms[1].boxes[1].name == "e"
    assert a.boxes[2].terms[1].boxes[2].name == "f"


@fixture
def localsum_testdata_monoidal():
    x, y, z = monoidal.Ty('x'), monoidal.Ty('y'), monoidal.Ty('z')
    f = monoidal.Box('f', x, y)
    g = monoidal.Box('g', x, y)
    h = monoidal.Box('h', y, z)
    diag = monoidal.LocalSum([f, g]) @ h
    diag = f @ (diag >> monoidal.Box("t", y @ z, x) >> f) @ g
    return diag


def test_distribute_tensor(localsum_testdata_monoidal):
    diag = localsum_testdata_monoidal.normal_form()
    a = str(diag)
    diag = distribute_tensor(diag, 1, 2)
    b = str(diag)
    assert isinstance(diag.boxes[1], monoidal.LocalSum)
    assert len(diag.boxes[1].terms) == 2
    assert len(diag.boxes[1].terms[0].boxes) == 2
    assert len(diag.boxes[1].terms[1].boxes) == 2
    assert diag.boxes[1].terms[0].boxes[0].name == "f"
    assert diag.boxes[1].terms[0].boxes[1].name == "h"
    assert diag.boxes[1].terms[0].boxes[0].name == "f"
    assert diag.boxes[1].terms[1].boxes[1].name == "h"
    assert len(diag.boxes[1].terms[0].layers.boxes) == 2
    assert len(diag.boxes[1].terms[1].layers.boxes) == 2
    assert diag.boxes[1].terms[0].offsets == [0,1]
    assert diag.boxes[1].terms[1].offsets == [0,1]

def test_check_nothing_in_way():
    x, y, z = monoidal.Ty('x'), monoidal.Ty('y'), monoidal.Ty('z')
    f = monoidal.Box('f', x, y)
    g = monoidal.Box('g', x, y)
    h = monoidal.Box('h', y, z)
    diag = monoidal.LocalSum([f, g]) @ h
    diag = diag >> (monoidal.Box("t", y, x) @ monoidal.Box("t2", z, x)) >> monoidal.Box('f', x @ x, y)
    assert _check_nothing_in_way(diag, 0, 2, [0])
    assert not _check_nothing_in_way(diag, 0, 3, [0])


def test_distribute_composition_monoidal_right():
    x, y, z = monoidal.Ty('x'), monoidal.Ty('y'), monoidal.Ty('z')
    f = monoidal.Box('f', x, y)
    g = monoidal.Box('g', x, y)
    h = monoidal.Box('h', y, z)
    diag = monoidal.LocalSum([f, g]) @ h
    diag = diag >> (monoidal.Box("t", y, x) @ monoidal.Box("t2", z, x)) >> monoidal.Box('f', x @ x, y)
    a = str(diag)
    diag = distribute_composition_monoidal(diag, 0, 2)
    b = str(diag)
    assert len(diag.boxes) == 4
    assert len(diag.boxes[0].terms) == 2
    assert len(diag.boxes[0].terms[0].boxes) == 2
    assert diag.boxes[0].terms[0].boxes[0].name == "f"
    assert diag.boxes[0].terms[0].boxes[1].name == "t"
    assert len(diag.boxes[0].terms[1].boxes) == 2
    assert diag.boxes[0].terms[1].boxes[0].name == "g"
    assert diag.boxes[0].terms[1].boxes[1].name == "t"


def test_distribute_composition_monoidal_left():
    x, y, z = monoidal.Ty('x'), monoidal.Ty('y'), monoidal.Ty('z')
    diag = monoidal.Box("a", x, y) >> monoidal.Box("b", y, x) >> monoidal.LocalSum([monoidal.Box("c", x, z), monoidal.Box("d", x, z)]) >> monoidal.Box("e", z, y)>> monoidal.Box("f", y, y)>> monoidal.Box("g", y, x)
    diag = distribute_composition_monoidal(diag, 2, 1)
    assert len(diag.boxes) == 5
    assert len(diag.boxes[1].terms) == 2
    assert len(diag.boxes[1].terms[0].boxes) == 2
    assert diag.boxes[1].terms[0].boxes[0].name == "b"
    assert diag.boxes[1].terms[0].boxes[1].name == "c"
    assert len(diag.boxes[1].terms[1].boxes) == 2
    assert diag.boxes[1].terms[1].boxes[0].name == "b"
    assert diag.boxes[1].terms[1].boxes[1].name == "d"
