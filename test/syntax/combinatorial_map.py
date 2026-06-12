import shutil

from pytest import raises

from discopy.combinatorial_map import (
    Permutation,
    port_side,
)
from discopy.compact import Box, Ty, CombinatorialMap as M, Hypergraph as H
from discopy.utils import AxiomError


def test_cycles():
    assert Permutation((1, 0, 3, 2)).cycles() == ((0, 1), (2, 3))
    assert Permutation.from_cycles([(0, 1), (2, 3)], 4) == (1, 0, 3, 2)
    assert Permutation((1, 0)).is_fixpoint_free_involution()
    assert not Permutation((0,)).is_fixpoint_free_involution()
    assert Permutation.identity(2) == (0, 1)
    assert Permutation((1, 0)).compose((1, 0)) == (0, 1)
    assert Permutation((1, 0)).tensor((1, 0)) == (1, 0, 3, 2)


def test_port_side():
    x = Ty('x')
    ports = M.id(x).ports
    assert port_side(ports[0]) == "left"
    assert port_side(ports[1]) == "right"


def test_default_compact_setting():
    x, y = map(Ty, "xy")
    f = Box('f', x, y)
    cmap = M.from_box(f)
    assert isinstance(f, M.category)
    assert cmap.to_hypergraph().category == M.category


def test_M_init():
    x, y, z = map(Ty, "xyz")
    f = Box('f', x @ y, z)
    valid = M.from_box(f)
    with raises(ValueError):
        M(x, x, (), ())
    with raises(ValueError):
        M(x, x, (), (0, 1))
    with raises(ValueError):
        M(x, x, (), (0,))
    with raises(AxiomError):
        M(x @ y, x @ y, (), (1, 0, 3, 2))
    with raises(ValueError):
        g = Box('g', x, x)
        M(g.dom, g.cod, (g, ), M.from_box(g).edge, (1, 0, 2, 3))
    assert M(
        f.dom, f.cod, (f, ), valid.edge, (0, 1, 4, 2, 3, 5)
    ).node_cycles == ((2, 4, 3), )
    with raises(ValueError):
        M(f.dom, f.cod, (f, ), valid.edge, tuple(range(valid.n_ports)))
    with raises(AxiomError):
        M(x, y, (), (1, 0))


def test_id_and_tensor():
    x, y = map(Ty, "xy")
    assert M.id(x).edge == (1, 0)
    assert M.id(x).node == (0, 1)
    assert M.id().tensor() == M.id()
    assert M.id(x).tensor(M.id(y)) == M.id(x) @ M.id(y)
    assert (M.id(x) @ M.id(y)).to_hypergraph() == H.id(x @ y)


def test_from_box_and_to_hypergraph():
    x, y = map(Ty, "xy")
    f = Box('f', x, y)
    cmap = M.from_box(f)
    assert cmap.edge == (1, 0, 3, 2)
    assert cmap.node == (0, 2, 1, 3)
    assert cmap.to_hypergraph() == f.to_hypergraph()


def test_scalar_box():
    s = Box('s', Ty(), Ty())
    cmap = M.from_box(s)
    assert cmap.edge == ()
    assert cmap.node == ()
    assert cmap.euler_characteristic == 1
    assert cmap.to_hypergraph() == s.to_hypergraph()


def test_from_hypergraph():
    x, y = map(Ty, "xy")
    f = Box('f', x, y).to_hypergraph()
    assert M.from_hypergraph(f).to_hypergraph() == f
    with raises(ValueError):
        M.from_hypergraph(H.spiders(1, 2, x))


def test_then():
    x, y, z, w = map(Ty, "xyzw")
    f, g, h = [M.from_box(box) for box in [
        Box('f', x, y), Box('g', y, z), Box('h', z, w)]]
    assert ((f >> g) >> h) == (f >> (g >> h))
    assert (f >> M.id(y)) == f
    assert (M.id(x) >> f) == f
    assert (f >> g).to_hypergraph() == f.to_hypergraph() >> g.to_hypergraph()
    with raises(AxiomError):
        f >> f


def test_tensor():
    x, y, z = map(Ty, "xyz")
    f = M.from_box(Box('f', x, y))
    g = M.from_box(Box('g', y, z))
    assert (f @ g).to_hypergraph() == f.to_hypergraph() @ g.to_hypergraph()
    assert (f @ M.id()) == f
    assert (M.id() @ f) == f


def test_tensor_then():
    x, y, z, a, b = map(Ty, "xyzab")
    f1 = M.from_box(Box('f1', x, y))
    f2 = M.from_box(Box('f2', y, z))
    g = M.from_box(Box('g', a, b))
    assert ((f1 >> f2) @ g).to_hypergraph() == (f1.to_hypergraph() >> f2.to_hypergraph()) @ g.to_hypergraph()


def test_then_tensor():
    x1, x2, y1, y2, z = map(Ty, ['x1', 'x2', 'y1', 'y2', 'z'])
    f1 = M.from_box(Box('f1', x1, y1))
    f2 = M.from_box(Box('f2', x2, y2))
    g = M.from_box(Box('g', y1 @ y2, z))
    assert ((f1 @ f2) >> g).to_hypergraph() == (f1.to_hypergraph() @ f2.to_hypergraph()) >> g.to_hypergraph()


def test_euler_characteristic():
    x, y = map(Ty, "xy")
    wire = M.id(x)
    box = M.from_box(Box('f', x, y))
    scalar = M.from_box(Box('s', Ty(), Ty()))
    assert wire.face_cycles == ((0, 1), )
    assert wire.euler_characteristic == 0
    assert box.face_cycles == ((0, 2, 3, 1), )
    assert box.euler_characteristic == 0
    assert (box @ scalar).euler_characteristic == 1

def test_draw_map(tmp_path):
    x, y = map(Ty, "xy")
    path = tmp_path / "map.dot"
    cmap = M.from_box(Box('f', x, y))
    dot = cmap.draw_map(seed=0)
    assert "graph combinatorial_map" in dot
    assert "layout=\"neato\"" in dot
    assert 'v0 [label="f", width="0.22", height="0.22"]' in dot
    assert "xlabel=\"(1 2)\"" not in dot
    assert "p1" not in dot
    assert 'b0 -- v0 [len="0.85", taillabel="0", headlabel="1"]' in dot
    assert cmap.draw_map(path=path, seed=0) == str(path)
    assert path.exists()
    assert path.read_text().startswith("graph combinatorial_map")
    if shutil.which("dot") is not None:
        png_path = tmp_path / "map.png"
        assert cmap.draw_map(path=png_path, engine="dot") == str(png_path)
        assert png_path.exists()
