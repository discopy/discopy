"""
A simple anywidget that displays a DisCoPy diagram as an SVG.

Example
-------
>>> from discopy.monoidal import Ty, Box
>>> f = Box('f', Ty('x'), Ty('y'))
>>> widget = f.to_widget()
>>> widget  # doctest: +SKIP
"""

try:
    import anywidget
    import traitlets
    _HAS_ANYWIDGET = True
except ImportError:  # pragma: no cover
    anywidget = None  # type: ignore
    traitlets = None  # type: ignore
    _HAS_ANYWIDGET = False


if _HAS_ANYWIDGET:

    class DiagramWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          el.style.cssText = `
            display: inline-block;
          `;
          let update = () => { el.innerHTML = model.get("svg") || ""; };
          model.on("change:svg", update);
          update();
        }
        export default { render };
        """
        _css = """
        .jupyter-widgets.anywidget-widget { display: inline-block; }
        """
        svg = traitlets.Unicode("").tag(sync=True)

else:

    class DiagramWidget:  # type: ignore
        def __init__(self, svg=""):
            raise ImportError(
                "anywidget is required for interactive widgets. "
                "Install it with: pip install anywidget"
            )
