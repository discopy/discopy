# -*- coding: utf-8 -*-

""" Discopy configuration. """

from discopy.utils import text_width

DEFAULT_BACKEND = 'numpy'
NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]


# Mapping from attribute to function from box to default value.
BOX_DRAWING_ATTRIBUTES = {
    "height": lambda _: 1,
    "is_conjugate": lambda _: False,
    "is_transpose": lambda _: False,
    "bubble_opening": lambda _: False,
    "bubble_closing": lambda _: False,
    "frame_boundary": lambda _: False,
    "frame_colour": lambda _: "lightgrey",
    "draw_as_braid": lambda _: False,
    "draw_as_cup": lambda _: False,
    "draw_as_cap": lambda _: False,
    "draw_as_dual_rail_braid": lambda _: False,
    "draw_as_dual_rail_twist": lambda _: False,
    "draw_as_dual_rail_cup": lambda _: False,
    "draw_as_dual_rail_cap": lambda _: False,
    "draw_as_wires": lambda box: any(getattr(box, a) for a in [
        "bubble_opening", "bubble_closing", "draw_as_braid",
        "draw_as_cup", "draw_as_cap"]),
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
    "no_label": lambda box: any([
        box.draw_as_wires, box.draw_as_spider, box.draw_as_brakets,
        box.draw_as_controlled, box.draw_as_discards, box.draw_as_measures,
        box.draw_as_dual_rail_braid, box.draw_as_dual_rail_twist,
        box.draw_as_dual_rail_cup, box.draw_as_dual_rail_cap]),
    "min_width": lambda box:
        0 if box.no_label else text_width(box.drawing_name),
    "tikzstyle_name": lambda box: (
        box.name if box.name.isidentifier() else "symbol")
}

WIRE_DRAWING_ATTRIBUTES = {
    "right_margin": lambda ob: text_width(str(ob)),
}

# Mapping from attribute to function from object to default value, for the
# coloured region on either side of a wire: ``min_right_margin`` adds extra
# space to the right of a wire, e.g. to make room for a long label, while
# ``ribbon`` is the region to the right of a wire whenever it carries a
# ``width``, i.e. the inside of a :class:`discopy.balanced.Ribbon` whose
# two rails are ``width`` apart, see :meth:`monoidal.Ty.wire_offsets`.
COLOUR_DRAWING_ATTRIBUTES = {
    "min_right_margin": lambda _: 0,
    "ribbon": lambda ob:
        ob.cod if hasattr(getattr(ob, "cod", None), "width") else None,
}

# Default drawing parameters.
DRAWING_DEFAULT = {
    "fontsize": 12,
    "margins": (0, 0),
    "textpad": (2**-4, 2**-4),
    "facecolor": "white",
    "edgecolor": "black",
    "use_tikzstyles": False,
    "braid_shadow": (.3, .1),
    # Legend width in inches is legend_base_width + legend_char_width
    # times the length of the longest label.
    "legend_base_width": 0.5,
    "legend_char_width": 0.085,
    # Gap in inches between the diagram and the legend.
    "legend_margin": 0.4,
    "ribbon_width": 0.25,
}

# Mapping from tikz colors to hexcodes.
COLORS = {
    "white": '#ffffff',
    "red": '#e8a5a5',
    "green": '#d8f8d8',
    "blue": '#776ff3',
    "yellow": '#f7f700',
    "gray": '#cccccc',
    "black": '#000000',
}


def darken(hexcode, factor=0.6):
    """ A darker shade of a hexcode, keeping ``factor`` of each RGB channel.

    Used to fill the back of a twisting ribbon with a darker shade of the
    colour filling its front, see
    :meth:`discopy.drawing.backend.Backend.draw_dual_rail_twist`.
    """
    return '#' + ''.join(f'{round(int(hexcode[i:i + 2], 16) * factor):02x}'
                         for i in (1, 3, 5))


# A key ``f"dark_{name}"`` for every named colour, mapping to a darker shade
# of its hexcode, see :func:`darken`.
COLORS.update({f"dark_{name}": darken(hexcode)
               for name, hexcode in COLORS.items()})

# The maximum depth of a ribbon cup or cap fold in the dual rail drawing. It
# caps the depth of the fold's half circle, so that a wide cup flattens into
# an ellipse rather than a deep semicircle and the drawing stays compact.
RIBBON_FOLD_DEPTH = 1.0

# Mapping from tikz shapes to matplotlib shapes.
SHAPES = {
    "rectangle": 's',
    "triangle_up": '^',
    "triangle_down": 'v',
    "circle": 'o',
    "plus": '+',
}
