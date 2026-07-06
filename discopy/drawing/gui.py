# -*- coding: utf-8 -*-

"""
An interactive GUI where each level of the hierarchy of diagrams is
implemented as a force-directed layout respecting its structure.

The GUI is a self-contained HTML page: the diagram is compiled to a layout
specification by :func:`drawing.layout.to_layout` and a force simulation
runs in the browser, with one degree of freedom for each feature of the
doctrine. For example, each box in a pivotal diagram is a pivot with its
ports fixed on a circle that can rotate around it, so that yanking the
snake is just letting the simulation relax.

The ladder at the top of the page lets one lay out the same diagram at any
higher level of the hierarchy, i.e. apply the inclusion functor and watch
the extra degrees of freedom kick in.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    to_html
    draw
    gallery

Example
-------
>>> from discopy.rigid import Ty, Box
>>> html = to_html(Box('f', Ty('x'), Ty('y')).transpose())
>>> assert "<canvas" in html and "pivot" in html
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from discopy.drawing.layout import HIERARCHY, to_layout

TEMPLATE = Path(__file__).parent / "gui.html"


def to_html(*diagrams, titles=None, blurbs=None) -> str:
    """
    Render diagrams as a self-contained HTML page with a force layout GUI.

    Parameters:
        diagrams : The diagrams to include in the gallery of the page.
        titles : Optional list of titles, one for each diagram.
        blurbs : Optional list of one-line descriptions.

    Example
    -------
    >>> from discopy.monoidal import Ty, Box
    >>> f = Box('f', Ty('x'), Ty('y'))
    >>> html = to_html(f, titles=["just a box"])
    >>> assert "just a box" in html
    """
    titles = titles or [f"diagram {i}" for i, _ in enumerate(diagrams)]
    blurbs = blurbs or len(diagrams) * [""]
    entries = [
        {"title": title, "blurb": blurb, "spec": to_layout(diagram)}
        for diagram, title, blurb in zip(diagrams, titles, blurbs)]
    hierarchy = [[name, sorted(features)] for name, features in HIERARCHY]
    return TEMPLATE.read_text(encoding="utf-8").replace(
        "__GALLERY_JSON__", json.dumps(entries)).replace(
        "__HIERARCHY_JSON__", json.dumps(hierarchy))


def draw(*diagrams, path=None, open_browser=False, **params) -> str:
    """
    Write the GUI page for the given diagrams and return its path.

    Parameters:
        diagrams : The diagrams to include in the gallery of the page.
        path : Where to write the page, a temporary file by default.
        open_browser : Whether to open the page in a web browser.
        params : Passed to :func:`to_html`.
    """
    html = to_html(*diagrams, **params)
    if path is None:
        with NamedTemporaryFile(
                mode='w', suffix=".html", prefix="discopy-gui-",
                delete=False) as file:
            file.write(html)
            path = file.name
    else:
        with open(path, 'w', encoding="utf-8") as file:
            file.write(html)
    if open_browser:  # pragma: no cover
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(path))
    return path


def gallery() -> tuple[list, list, list]:
    """
    Example diagrams for each level of the hierarchy, as a triple of
    diagrams, titles and blurbs to be passed to :func:`to_html`.

    >>> diagrams, titles, blurbs = gallery()
    >>> assert len(diagrams) == len(titles) == len(blurbs)
    """
    from discopy import (
        monoidal, braided, symmetric, traced, rigid, pivotal, ribbon,
        compact, frobenius)
    from discopy.drawing import spiral

    diagrams, titles, blurbs = [], [], []

    def add(diagram, title, blurb):
        diagrams.append(diagram)
        titles.append(title)
        blurbs.append(blurb)

    f0, f1 = (monoidal.Box(f"f{i}", monoidal.Ty(f"x{i}"),
                           monoidal.Ty(f"y{i}")) for i in (0, 1))
    g0, g1 = (monoidal.Box(f"g{i}", monoidal.Ty(f"y{i}"),
                           monoidal.Ty(f"z{i}")) for i in (0, 1))
    add(f0 @ f1 >> g0 @ g1, "interchanger",
        "Progressive and planar: wires flow down, boxes slide sideways.")
    add(spiral(2), "spiral",
        "The worst case for the deterministic layout, relaxed by springs.")

    a, b, c = map(braided.Ty, "abc")
    add(braided.Braid(a, b) @ c
        >> b @ braided.Braid(a, c)
        >> braided.Braid(b, c) @ a,
        "braid word",
        "Wires may cross over and under, but crossings cannot be undone.")

    u, v = map(traced.Ty, "uv")
    add(traced.Box('f', u @ v, u @ v).trace(),
        "feedback loop",
        "The trace lets a wire flow back up around the box.")

    p, q, r = map(symmetric.Ty, "pqr")
    add(symmetric.Swap(p, q) @ r
        >> q @ symmetric.Swap(p, r),
        "permutation",
        "Swaps cross wires freely: over and under no longer matter.")

    add(rigid.Box('f', rigid.Ty('x'), rigid.Ty('y')).transpose(),
        "transpose (rigid)",
        "Cups and caps bend wires, but boxes stay upright: "
        "the snake cannot be yanked.")

    add(pivotal.Box('f', pivotal.Ty('x'), pivotal.Ty('y')).transpose(),
        "transpose (pivotal)",
        "Each box is a pivot: drag the ring to rotate it, "
        "or let the torque yank the snake.")

    s = ribbon.Ty('s')
    add(ribbon.Twist(s) >> ribbon.Twist(s).dagger(),
        "twist", "Ribbons remember their framing: a twist and its inverse.")

    e = compact.Ty('e')
    add(compact.Cap(e.r, e)
        >> compact.Swap(e.r, e)
        >> compact.Cup(e, e.r),
        "unknot",
        "Compact: swaps and cups together untangle everything.")

    w = frobenius.Ty('w')
    add(frobenius.Spider(1, 3, w)
        >> w @ frobenius.Spider(2, 1, w)
        >> frobenius.Spider(2, 0, w),
        "spiders",
        "Frobenius spiders have unordered ports: "
        "wires attach at any angle.")

    return diagrams, titles, blurbs


def main(path=None):  # pragma: no cover
    """ Write the gallery page and print its path. """
    import sys
    path = path or (sys.argv[1] if len(sys.argv) > 1 else None)
    diagrams, titles, blurbs = gallery()
    print(draw(*diagrams, titles=titles, blurbs=blurbs, path=path))


if __name__ == "__main__":  # pragma: no cover
    main()
