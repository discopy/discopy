# -*- coding: utf-8 -*-

""" Discopy configuration. """

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]


def _box_label_width(box):
    # The width needed to fit the widest line of the box's name; drawing_name
    # is guaranteed to be set already by the "drawing_name" attribute below.
    from discopy.utils import text_width
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
