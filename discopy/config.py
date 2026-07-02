# -*- coding: utf-8 -*-

""" Discopy configuration. """

from functools import lru_cache

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

@lru_cache(maxsize=1024)
def text_width(text, fontsize=12):
    """ The width of a text label in drawing units, i.e. inches.

    Measured from the actual glyph outlines with matplotlib's text layout, so
    it is accurate for proportional fonts and for mathtext such as a LaTeX
    name (e.g. ``"$\\Lambda$"``). A drawing unit is one inch and a point is
    1/72 inch, hence the division of the point-sized text path.
    """
    if not text:
        return 0
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), text, size=fontsize).get_extents().width / 72


def box_label_width(box):
    """ The width needed to fit a box's name, in drawing units.

    This is the width of the widest line of the name (see :func:`text_width`),
    or zero if the box has no name.
    """
    name = getattr(box, "drawing_name", None)
    name = box.name if name is None else name
    if not name:
        return 0
    return max(text_width(line) for line in name.split("\n"))


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
    # Minimum width of the box outline, e.g. to fit a LaTeX name by hand.
    "min_width": lambda _: 0,
    # Depends on drawing_name, so it must come after it in this mapping.
    "box_label_width": box_label_width,
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
