# -*- coding: utf-8 -*-

""" Discopy configuration. """

from functools import lru_cache
from math import ceil

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

# A point is 1/72 inch; matplotlib measures text in points, drawings in inches.
POINTS_PER_INCH = 72

# Text widths are rounded up to a multiple of this resolution, a dyadic
# rational, so that widened boxes keep the same easy-to-read coordinates
# (integers plus or minus dyadic rationals) as the rest of a drawing.
TEXT_WIDTH_RESOLUTION = 2 ** -4


@lru_cache(maxsize=1024)
def text_width(text, fontsize=12):
    """ The width of a text label in drawing units, i.e. inches.

    Measured from the actual glyph outlines with matplotlib's text layout, so
    it is accurate for proportional fonts and for mathtext such as a LaTeX
    name (e.g. ``"$\\Lambda$"``). The result is rounded up to the nearest
    :data:`TEXT_WIDTH_RESOLUTION` so that it stays a dyadic rational, e.g.
    ``0.5625`` rather than ``0.5392252604166666``.
    """
    if not text:
        return 0
    from matplotlib.textpath import TextPath
    width = TextPath((0, 0), text, size=fontsize).get_extents().width
    width /= POINTS_PER_INCH
    return ceil(width / TEXT_WIDTH_RESOLUTION) * TEXT_WIDTH_RESOLUTION


def _box_label_width(box):
    # The width needed to fit the widest line of the box's name; drawing_name
    # is guaranteed to be set already by the "drawing_name" attribute below.
    return max(text_width(line) for line in box.drawing_name.split("\n"))


# Mapping from attribute to function from box to default value.
DRAWING_ATTRIBUTES = {
    "height": lambda _: 1,
    "is_conjugate": lambda _: False,
    "is_transpose": lambda _: False,
    "bubble_opening": lambda _: False,
    "bubble_closing": lambda _: False,
    "frame_boundary": lambda _: False,
    "draw_as_braid": lambda _: False,
    "draw_as_wires": lambda box: any(getattr(box, a) for a in [
        "bubble_opening", "bubble_closing", "draw_as_braid"]),
    "draw_as_spider": lambda _: False,
    "draw_as_brakets": lambda _: False,
    "draw_as_discards": lambda _: False,
    "draw_as_measures": lambda _: False,
    "draw_as_controlled": lambda _: False,
    "controlled": lambda _: None,  # Used for drawing controlled gates.
    "distance": lambda _: None,  # Used for drawing controlled gates.
    "_digits": lambda _: None,  # Used for drawing brakets.
    "shape": lambda box:
        "circle" if getattr(box, "draw_as_spider", False) else None,
    "color": lambda box:
        "black" if getattr(box, "draw_as_spider", False) else "white",
    "drawing_name": lambda box: box.name,
    # Whether the box is drawn as a rectangle with its name in the middle;
    # other shapes (wires, spiders, ...) draw themselves rather than a label.
    "draws_label": lambda box: not any((
        box.draw_as_wires, box.draw_as_spider, box.draw_as_brakets,
        box.draw_as_controlled, box.draw_as_discards, box.draw_as_measures)),
    # Minimum width of the box outline, e.g. to fit a LaTeX name by hand.
    "min_width": lambda _: 0,
    # Space needed to fit the box's name (depends on drawing_name above).
    "box_label_width": _box_label_width,
    # The width the outline must have to fit the name or the given min_width.
    "box_min_width": lambda box: max(
        box.box_label_width, box.min_width) if box.draws_label else 0,
    "tikzstyle_name": lambda box: (
        box.name if box.name.isidentifier() else "symbol")
}

# Default drawing parameters.
DRAWING_DEFAULT = {
    "fontsize": 12,
    "margins": (0, 0),
    "textpad": (2**-4, 2**-4),
    "facecolor": "white",
    "edgecolor": "black",
    "use_tikzstyles": False,
    "braid_shadow": (.3, .1)
}

# Mapping from tikz colors to hexcodes.
COLORS = {
    "white": '#ffffff',
    "red": '#e8a5a5',
    "green": '#d8f8d8',
    "blue": '#776ff3',
    "yellow": '#f7f700',
    "black": '#000000',
}

# Mapping from tikz shapes to matplotlib shapes.
SHAPES = {
    "rectangle": 's',
    "triangle_up": '^',
    "triangle_down": 'v',
    "circle": 'o',
    "plus": '+',
}
