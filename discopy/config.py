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
    """ The width needed to fit a box's name on one line, in drawing units.

    LaTeX math (any name containing a ``$``) is rendered by the backend, so
    its width cannot be guessed from the number of characters: we fall back to
    the default width and let :attr:`min_width` widen the box if needed.
    """
    name = getattr(box, "drawing_name", None)
    name = box.name if name is None else name
    if not name or "$" in name:
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
    "draw_as_dual_rail_braid": lambda _: False,
    "draw_as_dual_rail_twist": lambda _: False,
    "draw_as_dual_rail_cup": lambda _: False,
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


def darken(hexcode, factor=0.6):
    """ A darker shade of a hexcode, keeping ``factor`` of each RGB channel.

    Used to fill the back of a twisting ribbon with a darker shade of the
    colour filling its front, see
    :meth:`discopy.drawing.backend.Backend.draw_dual_rail_twist`.
    """
    channels = (int(hexcode[i:i + 2], 16) for i in (1, 3, 5))
    return '#' + ''.join(
        f'{round(channel * factor):02x}' for channel in channels)


# A key ``f"dark_{name}"`` for every named colour, mapping to a darker shade
# of its hexcode, see :func:`darken`.
COLORS.update({f"dark_{name}": darken(hexcode)
               for name, hexcode in COLORS.items()})

# Palette cycled through to fill the inside of ribbons in the dual rail drawing
# of balanced and ribbon diagrams, one colour per distinct object, see
# :meth:`discopy.balanced.Diagram.to_braided`.
RIBBON_COLORS = ("red", "green", "blue", "yellow")

# The vertical depth that a ribbon cup or cap folds by in the dual rail
# drawing. It caps the depth of the fold's half circle, so that a wide cup
# flattens into an ellipse rather than a deep semicircle and the drawing stays
# compact.
RIBBON_FOLD_DEPTH = 1.0

# Mapping from tikz shapes to matplotlib shapes.
SHAPES = {
    "rectangle": 's',
    "triangle_up": '^',
    "triangle_down": 'v',
    "circle": 'o',
    "plus": '+',
}
