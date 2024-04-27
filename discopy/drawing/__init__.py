""" DisCoPy's drawing modules. """

import os
from PIL import Image
from tempfile import NamedTemporaryFile, TemporaryDirectory

from discopy.utils import Node
from discopy.drawing import backend, drawing
from discopy.drawing.backend import (
    Backend,
    TikZ,
    Matplotlib,
    ATTRIBUTES,
    DEFAULT,
    COLORS,
    SHAPES,
)
from discopy.drawing.drawing import (
    Point,
    PlaneGraph,
    Drawing,
)


def needs_asymmetry(diagram):
    if hasattr(diagram, "terms"):
        return any(needs_asymmetry(d) for d in diagram.terms)
    return any(
        box.is_dagger
        or getattr(box, "is_conjugate", False)
        or getattr(box, "is_transpose", False)
        for box in diagram.boxes)


def draw(diagram, **params):
    """
    Draws a diagram using networkx and matplotlib.

    Parameters
    ----------
    draw_as_nodes : bool, optional
        Whether to draw boxes as nodes, default is :code:`False`.
    color : string, optional
        Color of the box or node, default is white (:code:`'#ffffff'`) for
        boxes and red (:code:`'#ff0000'`) for nodes.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.1)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is :code:`False`.
    draw_box_labels : bool, optional
        Whether to draw box labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['auto', 'equal']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    nodesize : float, optional
        Node size for spiders and controlled gates.
    fontsize : int, optional
        Font size for the boxes, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call :code:`plt.show()`.
    to_tikz : bool, optional
        Whether to output tikz code instead of matplotlib.
    asymmetry : float, optional
        Make a box and its dagger mirror images, default is
        :code:`.25 * any(box.is_dagger for box in diagram.boxes)`.
    """
    params['asymmetry'] = params.get(
        'asymmetry', .25 * needs_asymmetry(diagram))
    return diagram.to_drawing().draw(**params)


def to_gif(diagram, *diagrams, **params):  # pragma: no cover
    """
    Builds a gif with the normalisation steps.

    Parameters
    ----------
    diagrams : :class:`Diagram`, optional
        Sequence of diagrams to draw.
    path : str
        Where to save the image, if :code:`None` a gif gets created.
    timestep : int, optional
        Time step in milliseconds, default is :code:`500`.
    loop : bool, optional
        Whether to loop, default is :code:`False`
    params : any, optional
        Passed to :meth:`Diagram.draw`.
    """
    path = params.pop("path", None)
    timestep = params.get("timestep", 500)
    loop = params.get("loop", False)
    steps, frames = (diagram, ) + diagrams, []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)
    with TemporaryDirectory() as directory:
        for i, _diagram in enumerate(steps):
            tmp_path = os.path.join(directory, f'{i}.png')
            _diagram.draw(path=tmp_path, **params)
            frames.append(Image.open(tmp_path))
        if loop:
            frames = frames + frames[::-1]
        frames[0].save(path, format='GIF', append_images=frames[1:],
                       save_all=True, duration=timestep,
                       **{'loop': 0} if loop else {})
        try:
            from IPython.display import HTML
            return HTML(f'<img src="{path}">')
        except ImportError:
            return f'<img src="{path}">'


class Equation:
    """
    An equation is a list of diagrams with a dedicated draw method.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between the terms.
        space : The space between the terms.

    Example
    -------
    >>> from discopy.tensor import Spider, Swap, Dim, Id
    >>> dim = Dim(2)
    >>> mu, eta = Spider(2, 1, dim), Spider(0, 1, dim)
    >>> delta, upsilon = Spider(1, 2, dim), Spider(1, 0, dim)
    >>> special = Equation(mu >> delta, Id(dim))
    >>> frobenius = Equation(
    ...     delta @ Id(dim) >> Id(dim) @ mu,
    ...     mu >> delta,
    ...     Id(dim) @ delta >> mu @ Id(dim))
    >>> Equation(special, frobenius, symbol=', ').draw(
    ...          aspect='equal', draw_type_labels=False, figsize=(8, 2),
    ...          path='docs/_static/drawing/frobenius-axioms.png')

    .. image:: /_static/drawing/frobenius-axioms.png
        :align: center
    """
    def __init__(self, *terms: "monoidal.Diagram", symbol="=", space=1):
        self.terms, self.symbol, self.space = terms, symbol, space

    def __repr__(self):
        return f"Equation({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f" {self.symbol} ".join(map(str, self.terms))

    def draw(self, path=None, space=None, **params):
        """
        Drawing an equation.

        Parameters:
            path : Where to save the drawing.
            space : The amount of space between the terms.
            params : Passed to :meth:`discopy.monoidal.Diagram.draw`.
        """
        def height(term):
            # i.e. if isinstance(diagram, (Sum, Equation))
            if hasattr(term, "terms"):
                return max(height(d) for d in term.terms)
            return len(term) or 1

        params['asymmetry'] = params.get(
            'asymmetry', .25 * needs_asymmetry(self))
        space = space or self.space
        max_height = max(map(height, self.terms))
        pad = params.get('pad', (0, 0))
        scale_x, scale_y = params.get('scale', (1, 1))
        backend = params['backend'] if 'backend' in params\
            else TikZ(
                use_tikzstyles=params.get('use_tikzstyles', None))\
            if params.get('to_tikz', False)\
            else Matplotlib(figsize=params.get('figsize', None))

        for i, term in enumerate(self.terms):
            scale = (scale_x, scale_y * max_height / height(term))
            term.draw(**dict(
                params, show=False, path=None,
                backend=backend, scale=scale, pad=pad))
            pad = (backend.max_width + space, 0)
            if i < len(self.terms) - 1:
                backend.draw_text(
                    self.symbol, pad[0], scale_y * max_height / 2)
                pad = (pad[0] + space, pad[1])

        return backend.output(
            path=path,
            baseline=max_height / 2,
            tikz_options=params.get('tikz_options', None),
            show=params.get("show", True),
            margins=params.get('margins', DEFAULT['margins']),
            aspect=params.get('aspect', DEFAULT['aspect']))

    def __bool__(self):
        return all(term == self.terms[0] for term in self.terms)


def spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    from discopy.monoidal import Ty, Box, Id
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    for box in [unit, counit, cup, cap]:
        box.draw_as_spider, box.drawing_name = True, ""
        box.shape, box.color = "circle", "black"
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result
