""" DisCoPy's drawing modules. """

import os
from PIL import Image
from tempfile import NamedTemporaryFile, TemporaryDirectory

from discopy.utils import Node, Point
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
    Equation,
)


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
    wire_labels : bool, optional
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
    steps, frames = [d.to_drawing() for d in (diagram, ) + diagrams], []
    path = path or os.path.basename(NamedTemporaryFile(
        suffix='.gif', prefix='tmp_', dir='.').name)
    if 'figsize' not in params:
        params['figsize'] = tuple(
            max(getattr(step, attr) for step in steps)
            for attr in ("width", "height"))
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
