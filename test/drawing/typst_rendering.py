"""Tests for the Typst/CeTZ drawing backend."""

import os

import pytest

from discopy.monoidal import Ty, Box
from discopy.symmetric import Swap
from discopy.rigid import Ty as RTy, Cup, Cap, Id
from discopy.braided import Braid
from discopy.frobenius import Ty as FTy, Spider
from discopy.quantum.zx import Z, X, H

TYPST_FOLDER = "test/drawing/typst/"


def _normalize_typst(text):
    """Strip trailing whitespace from each line for comparison."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def typst_and_compare(file):
    """Decorator that compares to_typst() output against a reference."""
    def decorator(func):
        def wrapper():
            diagram = func()
            source = diagram.to_typst().render()
            true_path = os.path.join(TYPST_FOLDER, file)
            if not os.path.exists(true_path):
                with open(true_path, "w") as f:
                    f.write(source)
                return
            with open(true_path) as f:
                assert _normalize_typst(source) == _normalize_typst(f.read())
        return wrapper
    return decorator


@typst_and_compare("box.typ")
def test_box_to_typst():
    x, y = Ty("x"), Ty("y")
    return Box("f", x, y)


@typst_and_compare("composition.typ")
def test_composition_to_typst():
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    g = Box("g", y, x)
    return f >> g


@typst_and_compare("tensor.typ")
def test_tensor_to_typst():
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    g = Box("g", y, x)
    return f @ g


@typst_and_compare("swap.typ")
def test_swap_to_typst():
    x = Ty("x")
    return Swap(x, x)


@typst_and_compare("braid.typ")
def test_braid_to_typst():
    x = Ty("x")
    return Braid(x, x)


@typst_and_compare("cup.typ")
def test_cup_to_typst():
    n = RTy("n")
    return Cup(n, n.r)


@typst_and_compare("cap.typ")
def test_cap_to_typst():
    n = RTy("n")
    return Cap(n.r, n)


@typst_and_compare("spider.typ")
def test_spider_to_typst():
    x = FTy("x")
    # Explicit colour: the `unfuse` doctest sets `Spider.color` globally.
    return Spider(2, 1, x, color="black")


@typst_and_compare("snake_equation.typ")
def test_snake_equation_to_typst():
    n = RTy("n")
    return Id(n.r) >> Cap(n.r, n) @ Id(n.r) >> Id(n.r) @ Cup(n, n.r)


@typst_and_compare("zx_z_spider.typ")
def test_zx_z_to_typst():
    return Z(2, 1)


@typst_and_compare("zx_x_spider.typ")
def test_zx_x_to_typst():
    return X(2, 1)


@typst_and_compare("hadamard.typ")
def test_hadamard_to_typst():
    return H


def test_to_typst_returns_document():
    """to_typst() returns a Document AST with correct structure."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    doc = f.to_typst()
    source = doc.render()
    assert 'import "@preview/cetz' in source
    assert 'canvas(' in source
    assert "import draw: *" in source


def test_to_typst_includes_mathematical_labels():
    """Box labels with LaTeX syntax remain intact."""
    x, y = Ty("x"), Ty("y")
    f = Box("$f$", x, y)
    source = f.to_typst().render()
    assert "[$f$]" in source


def test_typst_source_is_deterministic():
    """Multiple calls give identical output."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    assert f.to_typst().render() == f.to_typst().render()


def test_typst_backend_in_drawing_init():
    """The Typst backend is exported from discopy.drawing."""
    from discopy.drawing import Typst
    assert Typst is not None


def test_typst_compilation():
    """Compilation to SVG works when typst-py is installed."""
    try:
        import typst  # noqa: F401
    except ImportError:
        pytest.skip("typst package not installed")
    from discopy.monoidal import Ty, Box
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    svg = f.draw(format="typst", show=False)
    assert isinstance(svg, bytes)
    assert svg.startswith(b"<svg") or svg.startswith(b"<?xml")


def test_typst_output_to_file(tmp_path):
    """``output`` writes Typst source for ``.typ`` and SVG bytes otherwise."""
    x, y = Ty("x"), Ty("y")
    f = Box("f", x, y)
    source = f.draw(format="typst", path=tmp_path / "f.typ", show=False)
    assert (tmp_path / "f.typ").read_text() == source
    assert 'import "@preview/cetz' in source
    svg = f.draw(format="typst", path=tmp_path / "f.svg", show=False)
    assert (tmp_path / "f.svg").read_bytes() == svg


def test_typst_colors_and_labels():
    """Colours, transparent fills and spider labels reach the CeTZ source."""
    x = FTy("x")
    # A phase gives the spider a label, drawn white on black, black on red.
    dark = Spider(2, 1, x, 0.5, color="black").to_typst().render()
    assert 'fill: rgb("#ffffff")' in dark
    light = Spider(2, 1, x, 0.5, color="red").to_typst().render()
    assert 'rgb("#e8a5a5")' in light
    assert 'fill: rgb("#000000")' in light
    transparent = Box("f", x, x, color="none").to_typst().render()
    assert "fill: none" in transparent


def test_typst_color_helpers():
    """``format_color`` and ``color_expr`` handle names, hexcodes and none."""
    from discopy.drawing.backend import Typst
    assert Typst.format_color(None) == "none"
    assert Typst.format_color("#123456") == "#123456"
    assert Typst.format_color("red") == "#e8a5a5"
    assert Typst.color_expr("none") == "none"
    assert Typst.color_expr("#123456") == 'rgb("#123456")'
    assert Typst.color_expr("white") == "white"


def test_wire_bezier_points():
    """A wire only bends when it is neither horizontal nor vertical."""
    from discopy.drawing.backend import wire_bezier_points
    assert wire_bezier_points((0, 0), (0, 1), True, True) is None
    assert wire_bezier_points((0, 0), (1, 0), True, True) is None
    assert wire_bezier_points((0, 0), (1, 1), False, False) is None
    both = wire_bezier_points((0, 0), (3, 3), True, True)
    assert both == wire_bezier_points((0, 0), (3, 3), True, False)
    assert both == ((2, 0), (3, 1))
    assert wire_bezier_points((0, 0), (3, 3), False, True) == ((0, 2), (1, 3))


def test_typst_fontsize_and_text_colour():
    """A non-default fontsize and a coloured box name are honoured."""
    x, y = Ty("x"), Ty("y")
    source = Box("f", x, y).to_typst(fontsize=24).render()
    assert "text(" in source and "size:" in source


def test_typst_ast_nodes():
    """The AST nodes render the Typst syntax they stand for."""
    from discopy.drawing.typst_ast import Import, TypstNode
    assert Import("cetz", members=["canvas"]).render() \
        == '#import "cetz": canvas'
    assert Import("fletcher", alias="f").render() == '#import "fletcher" as f'
    assert Import("cetz").render() == '#import "cetz"'
    with pytest.raises(NotImplementedError):
        TypstNode().render()


def test_typst_show_returns_ipython_svg():
    """With no path and ``show``, ``output`` hands IPython an SVG to display."""
    pytest.importorskip("IPython")
    x, y = Ty("x"), Ty("y")
    svg = Box("f", x, y).draw(format="typst", show=True)
    assert type(svg).__name__ == "SVG"
