# -*- coding: utf-8 -*-

""" Discopy configuration. """

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

# Mapping from attribute to function from box to default value.
DRAWING_ATTRIBUTES = {
    "draw_as_braid": lambda _: False,
    "draw_as_wires": lambda box: box.draw_as_braid,
    "draw_as_spider": lambda _: False,
    "draw_as_brakets": lambda _: False,
    "draw_as_discards": lambda _: False,
    "draw_as_measures": lambda _: False,
    "draw_as_controlled": lambda _: False,
    "shape": lambda box:
        "circle" if getattr(box, "draw_as_spider", False) else None,
    "color": lambda box:
        "red" if getattr(box, "draw_as_spider", False) else "white",
    "drawing_name": lambda box: box.name,
    "tikzstyle_name": lambda box: box.name,
}

# Default drawing parameters.
DRAWING_DEFAULT = {
    "aspect": "auto",
    "fontsize": 12,
    "margins": (.05, .1),
    "textpad": (.1, .1),
    "color": 'white',
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
