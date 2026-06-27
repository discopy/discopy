# -*- coding: utf-8 -*-

""" Discopy configuration. """

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

# Width (in drawing units) of one monospace character at the default fontsize.
# Box labels are drawn in a monospace font so that the width needed to fit a
# name is simply its number of characters times this coefficient. A drawing
# unit is one inch, a point is 1/72 inch and a monospace glyph is about 0.6 em
# wide, hence the coefficient fontsize / 72 * 0.6 = fontsize / 120.
BOX_LABEL_CHAR_WIDTH = 12 / 120


def box_label_width(box):
    """ The width needed to fit a box's name on one line, in drawing units. """
    name = getattr(box, "drawing_name", None)
    name = box.name if name is None else name
    if not name:
        return 0
    longest_line = max(name.split("\n"), key=len)
    return len(longest_line) * BOX_LABEL_CHAR_WIDTH


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
