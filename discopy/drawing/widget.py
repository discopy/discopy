"""
An `anywidget <https://anywidget.dev>`_ for the rich display of diagrams.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    DiagramWidget

Note
----
This module requires the optional dependency ``anywidget``, i.e. it is only
imported by :meth:`discopy.utils.RichDisplay.to_widget`.

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> f = Box('f', Ty('x'), Ty('y'))
>>> widget = f.to_widget()
>>> assert isinstance(widget, DiagramWidget)
>>> assert widget.svg == f.to_svg()
"""

import anywidget
import traitlets


class DiagramWidget(anywidget.AnyWidget):
    """
    A widget rendering the SVG of a diagram, see
    :meth:`discopy.utils.RichDisplay.to_widget`.

    Parameters:
        svg : The source of the image, synced with the frontend.
    """
    svg = traitlets.Unicode("").tag(sync=True)

    _esm = """
    function render({ model, el }) {
      el.classList.add("discopy-diagram");
      const update = () => { el.innerHTML = model.get("svg"); };
      model.on("change:svg", update);
      update();
    }
    export default { render };
    """
    _css = """
    .discopy-diagram > svg { max-width: 100%; height: auto; }
    """
