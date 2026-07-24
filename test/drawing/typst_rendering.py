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
    return Spider(2, 1, x)


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
