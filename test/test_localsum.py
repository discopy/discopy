# -*- coding: utf-8 -*-

from pytest import raises, fixture
from discopy.cat import *
from discopy.localsum_rewriting import *


@fixture
def localsum_testdata_cat():
    x, y, z = Ob('x'), Ob('y'), Ob('z')
    f = Box('f', x, y)
    g = Box('g', x, y)
    h = Box('h', y, z)
    fg = LocalSum([f, g])
    k = fg >> h
    l = Box("h'", z, x) >> fg
    return x, y, z, f, g, h, fg, k, l


def test_localsum_creation(localsum_testdata_cat):
    _, _, _, _, _, _, _, k, l = localsum_testdata_cat
    assert isinstance(k.boxes[0], LocalSum)
    assert isinstance(l.boxes[1], LocalSum)


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
    diag = Box("a", x, y) >> Box("b", y, x) >> LocalSum([Box("c", x, z), Box(
        "d", x, z)]) >> Box("e", z, y) >> Box("f", y, y) >> Box("g", y, x)
    a = distribute_composition_cat(diag, 2, 1)
    assert len(a.boxes) == 5
    assert isinstance(a.boxes[1], LocalSum)
    assert a.boxes[1].terms[0].boxes[0].name == "b"
    a = distribute_composition_cat(diag, 2, 4)
    assert len(a.boxes) == 4
    assert isinstance(a.boxes[2], LocalSum)
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
