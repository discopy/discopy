
import os

from pytest import raises

from discopy.utils import AxiomError
from discopy.config import DRAWING_DEFAULT
from discopy.compact import *
from discopy.drawing import *
from discopy import config, monoidal

TIKZ_FOLDER = 'test/drawing/tikz/'


def tikz_and_compare(file, folder=TIKZ_FOLDER, **params):
    def decorator(func):
        def wrapper():
            diagram = func()
            draw = params.get('draw', type(diagram).draw)
            true_paths = [os.path.join(folder, file)]
            test_paths = [os.path.join(folder, '_' + file)]
            if params.get("use_tikzstyles", DRAWING_DEFAULT['use_tikzstyles']):
                true_paths.append(
                    true_paths[0].replace('.tikz', '.tikzstyles'))
                test_paths.append(
                    test_paths[0].replace('.tikz', '.tikzstyles'))
            draw(diagram, path=test_paths[0], **dict(params, to_tikz=True))
            for true_path, test_path in zip(true_paths, test_paths):
                if not os.path.exists(true_path):
                    os.replace(test_path, true_path)
                    continue
                with open(true_path, "r") as true:
                    with open(test_path, "r") as test:
                        assert true.read() == test.read()
                os.remove(test_path)
        return wrapper
    return decorator


def test_draw_baseline(tmp_path, monkeypatch):
    path = tmp_path / "box.svg"
    actual_path = tmp_path / "_box.svg"
    box = Box("f", Ty("x"), Ty("x"))

    box.draw(doctest=path, show=False)
    box.draw(doctest=path, show=False)
    assert not actual_path.exists()

    path.write_text("<svg/>")
    with raises(ValueError, match="Drawing differs"):
        box.draw(doctest=path, show=False)
    assert actual_path.exists()

    # Deleting a failing baseline regenerates it on the next draw.
    path.unlink()
    box.draw(doctest=path, show=False)
    assert path.exists()

    monkeypatch.setattr(config, "OVERRIDE_DOCTEST_IMAGES", True)
    path.write_text("<svg/>")
    box.draw(doctest=path, show=False)
    assert path.read_text() != "<svg/>"

    # A plain path just saves the drawing, overwriting silently.
    monkeypatch.setattr(config, "OVERRIDE_DOCTEST_IMAGES", False)
    path.write_text("<svg/>")
    box.draw(path=path, show=False)
    assert path.read_text() != "<svg/>"


def test_svg_equal(tmp_path):
    expected = tmp_path / "expected.svg"
    actual = tmp_path / "actual.svg"
    template = """\
<svg xmlns="http://www.w3.org/2000/svg" width="{width}">
  <g id="{name}"><text x="1">f</text></g>
</svg>"""
    expected.write_text(template.format(width=1, name="one"))

    # Rounding errors within the tolerance are forgiven.
    actual.write_text(template.format(width=1.5, name="one"))
    assert backend.svg_equal(expected, actual)

    # A genuine difference in width, position or text content is preserved.
    actual.write_text(template.format(width=9, name="one"))
    assert not backend.svg_equal(expected, actual)

    # A non-numeric difference, e.g. in an identifier, is also preserved.
    actual.write_text(template.format(width=1, name="two"))
    assert not backend.svg_equal(expected, actual)


def test_compare_drawing_raster_and_bytes(tmp_path):
    from PIL import Image
    baseline, actual = tmp_path / "box.png", tmp_path / "_box.png"
    Image.new("RGB", (8, 8), "white").save(baseline)
    Image.new("RGB", (8, 8), "white").save(actual)
    backend.compare_drawing(baseline, actual)
    assert not actual.exists()

    Image.new("RGB", (8, 8), "black").save(actual)
    with raises(ValueError, match="Drawing differs"):
        backend.compare_drawing(baseline, actual)

    baseline, actual = tmp_path / "box.tikz", tmp_path / "_box.tikz"
    baseline.write_text("tikz")
    actual.write_text("tikz")
    backend.compare_drawing(baseline, actual)
    assert not actual.exists()


def test_draw_coloured_regions_and_frame():
    red, green, blue = map(
        monoidal.Colour, ("red", "green", "blue"))
    x = monoidal.Ty(monoidal.Wire("x", red, green))
    y = monoidal.Ty(monoidal.Wire("y", green, blue))
    z = monoidal.Ty(monoidal.Wire("z", red, blue))
    box = monoidal.Box("f", x @ y, z)
    outer = monoidal.Ty(monoidal.Wire("u", blue, red))
    # A box fills its three wire regions, with the names in
    # discopy.config.COLORS resolved to their hexcodes as for boxes.
    assert {'#e8a5a5', '#d8f8d8', '#776ff3'} <= region_hexes(box)
    # A frame additionally fills its frame background (lightgrey).
    frame = box.bubble(dom=outer, cod=outer, draw_as_frame=True)
    assert {'#e8a5a5', '#d8f8d8', '#776ff3', '#d3d3d3'}\
        <= region_hexes(frame)


def coloured_bubble():
    """
    A bubble whose ten planar regions each get a distinct colour: six
    outside (left, two along the top, right, two along the bottom) and
    four inside (left, above and below the inner box, right). Every region
    is enclosed by wires, so all ten colours show only when the bubble's
    top and bottom boundaries are drawn, see issue #426.
    """
    Ty, Wire, Colour = monoidal.Ty, monoidal.Wire, monoidal.Colour
    ol, o1, o2, o_r, o3, o4 = map(Colour, (
        "red", "orange", "gold", "green", "blue", "purple"))
    il, i1, i_r, i2 = map(Colour, ("cyan", "magenta", "brown", "pink"))
    outer_dom = Ty(Wire("d", ol, o1), Wire("c", o1, o2), Wire("c", o2, o_r))
    outer_cod = Ty(Wire("b", ol, o3), Wire("a", o3, o4), Wire("a", o4, o_r))
    inner_dom = Ty(Wire("a", il, i1), Wire("b", i1, i_r))
    inner_cod = Ty(Wire("c", il, i2), Wire("d", i2, i_r))
    return monoidal.Box("f", inner_dom, inner_cod).bubble(
        dom=outer_dom, cod=outer_cod, name="g")


def test_bubble_regions_are_distinct():
    # All ten regions get their own colour only when the bubble's top and
    # bottom boundaries enclose the four inside regions, see issue #426.
    assert len(region_hexes(coloured_bubble())) == 10


def test_bubble_boundary_is_visible():
    # A plain bubble opening keeps its horizontal boundary, i.e. its box
    # node is not a frame side, while the frame sides of a square slot are.
    x, y, z = map(monoidal.Ty, "xyz")
    box_node, = Drawing.frame_opening(x, y, z, monoidal.Ty("")).box_nodes
    assert not Backend.is_frame_boundary(box_node)
    slot = Drawing.from_box(
        monoidal.Box("f", x, x)).slot(monoidal.Colour("white"))
    frame_box_nodes = [n for n in slot.box_nodes if n.box.frame_boundary]
    assert frame_box_nodes
    assert all(map(Backend.is_frame_boundary, frame_box_nodes))


def region_hexes(diagram, **params):
    """The set of region facecolours (as hex) drawn for a diagram."""
    from matplotlib.colors import to_hex
    from matplotlib import pyplot as plt
    drawing = diagram.to_drawing()
    drawing.add_box_corners()
    backend = Matplotlib(figsize=(2, 2))
    backend.draw_regions(drawing, **params)
    hexes = {to_hex(patch.get_facecolor()) for patch in backend.axis.patches}
    plt.close(backend.axis.figure)
    return hexes


def test_draw_regions_uncoloured_shapes():
    # Region filling runs for cups, caps, swaps, spiders and many-legged
    # boxes; with no colours every region is the default white.
    from discopy.frobenius import Spider, Ty as FTy
    x = Ty('x')
    shapes = [
        Cup(x, x.r), Cap(x.r, x), Swap(x, x),
        Box('f', x @ x, x @ x @ x), Spider(2, 1, FTy('x')),
        Cap(x.r, x) >> Swap(x.r, x) >> Cup(x, x.r)]
    for shape in shapes:
        assert region_hexes(shape) == {'#ffffff'}


def test_draw_coloured_cups_and_caps():
    red, green = map(monoidal.Colour, ("red", "green"))
    x = Ty(Ob("x", dom=red, cod=green))
    # A cup and a cap each separate the two boundary regions, with the
    # names in discopy.config.COLORS resolved to their hexcodes.
    assert region_hexes(Cup(x, x.r)) == {'#e8a5a5', '#d8f8d8'}
    assert region_hexes(Cap(x.r, x)) == {'#e8a5a5', '#d8f8d8'}


def test_draw_coloured_crossings_are_monochrome():
    from discopy.frobenius import Spider, Ty as FTy, Ob as FOb
    red = monoidal.Colour("red")
    # Wires that cross or merge must be globular, i.e. carry the same colour
    # on both sides, so their regions are a single colour.
    assert region_hexes(Swap(
        Ty(Ob("x", dom=red, cod=red)), Ty(Ob("y", dom=red, cod=red)))
    ) == {'#e8a5a5'}
    assert region_hexes(Spider(2, 1, FTy(FOb("x", dom=red, cod=red)))) == {
        '#e8a5a5'}
    # A swap of wires separating different regions is not globular.
    green = monoidal.Colour("green")
    with raises(AxiomError):
        Swap(Ty(Ob("x", dom=red, cod=green)), Ty(Ob("y", dom=green, cod=red)))


def test_draw_coloured_equation():
    red, green = map(monoidal.Colour, ("red", "green"))
    x = Ty(Ob("x", dom=red, cod=green))
    equation = Equation(Box("f", x, x), Box("g", x, x))
    colours = region_hexes(equation)
    # Both term regions show, each in its own white-bordered slot.
    assert {'#e8a5a5', '#d8f8d8', '#ffffff'} <= colours


def test_draw_region_non_colors_string():
    # Colours need not be discopy COLORS keys: any Matplotlib colour string
    # (a CSS name or a hex code) is filled as given.
    for name, hexcode in [("lightgrey", '#d3d3d3'), ("#abcdef", '#abcdef')]:
        c = monoidal.Colour(name)
        box = monoidal.Box("f", monoidal.Ty(monoidal.Wire("x", c, c)),
                           monoidal.Ty(monoidal.Wire("x", c, c)))
        assert hexcode in region_hexes(box)


def test_draw_legend():
    from matplotlib.colors import to_hex
    from matplotlib import pyplot as plt
    red, green, blue = map(monoidal.Colour, ("red", "green", "blue"))
    x = monoidal.Ty(monoidal.Wire("x", red, green))
    y = monoidal.Ty(monoidal.Wire("y", green, blue))
    z = monoidal.Ty(monoidal.Wire("z", red, blue))
    drawing = monoidal.Box("f", x @ y, z).to_drawing()
    drawing.add_box_corners()
    backend = Matplotlib(figsize=(3, 3))
    backend.draw_regions(drawing)
    backend.draw_legend(drawing)
    legend = backend.axis.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert set(labels) == {"red", "green", "blue"}
    # Each swatch is filled with its own colour, white is left out.
    swatches = {to_hex(handle.get_facecolor())
                for handle in legend.legend_handles}
    assert swatches == {'#e8a5a5', '#d8f8d8', '#776ff3'}
    plt.close(backend.axis.figure)


def test_draw_legend_skipped_without_colours():
    from matplotlib import pyplot as plt
    drawing = Box("f", Ty("a"), Ty("a")).to_drawing()
    drawing.add_box_corners()
    backend = Matplotlib(figsize=(2, 2))
    backend.draw_legend(drawing)
    assert backend.axis.get_legend() is None
    plt.close(backend.axis.figure)


def test_draw_legend_uses_colour_label():
    from matplotlib.colors import to_hex
    from matplotlib import pyplot as plt
    # A label gives the region a name in the legend while filling with its
    # actual colour.
    a = monoidal.Colour("cornflowerblue", label="Function")
    b = monoidal.Colour("palegreen", label="Morphism")
    x = monoidal.Ty(monoidal.Wire("F", dom=a, cod=b))
    drawing = monoidal.Box("f", x, x).to_drawing()
    drawing.add_box_corners()
    backend = Matplotlib(figsize=(3, 3))
    backend.draw_regions(drawing)
    backend.draw_legend(drawing)
    legend = backend.axis.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Function", "Morphism"]
    assert sorted(to_hex(handle.get_facecolor())
                  for handle in legend.legend_handles) == ['#6495ed', '#98fb98']
    plt.close(backend.axis.figure)


def test_draw_legend_figsize_and_space():
    import tempfile
    from matplotlib import image as mpimg
    red, green = monoidal.Colour("red"), monoidal.Colour("green")
    x = monoidal.Ty(monoidal.Wire("x", red, green))
    box = monoidal.Box("f", x, x)
    with tempfile.TemporaryDirectory() as folder:
        plain = os.path.join(folder, "plain.png")
        legend = os.path.join(folder, "legend.png")
        box.draw(show=False, figsize=(3, 2), path=plain)
        # With an explicit figsize the figure is widened by legend_space.
        box.draw(show=False, figsize=(3, 2), legend=True, legend_space=2,
                 path=legend)
        assert mpimg.imread(legend).shape[1] > mpimg.imread(plain).shape[1]
    # legend=True on an uncoloured diagram adds nothing.
    Box("g", Ty("a"), Ty("a")).draw(show=False, legend=True)


def test_draw_right_region_example():
    """
    Concrete example clarifying ``Matplotlib._draw_right_region`` and the
    ``Backend.draw_curved_polygon`` primitive it is built on: the curved
    polygon filling the region to the right of a wire, up to the diagram's
    right-hand edge.

    Consider a wire leaving a box at its top-right corner (0, 1) and
    bending down to (1, 0) (``bend_out=True``), inside a diagram of
    ``width=2``. The region to its right is the curved quadrilateral:
        * (0, 1) -- ``source``, where the wire leaves the box;
        * (1, 1) -- the Bezier control point, level with the source and
          plumb with the target, so the curve hugs the bend;
        * (1, 0) -- ``target``, where the wire is drawn to next;
        * (2, 0) -- straight across to the diagram's right-hand edge;
        * (2, 1) -- straight up along the right-hand edge;
        * back to (0, 1), closing the polygon.
    """
    from matplotlib import pyplot as plt
    from matplotlib.path import Path
    backend = Matplotlib(figsize=(2, 2))
    backend._draw_right_region(
        (0, 1), (1, 0), width=2, facecolor="red", bend_out=True)
    path = backend.axis.patches[-1].get_path()
    assert [tuple(vertex) for vertex in path.vertices] == [
        (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (0, 1)]
    assert list(path.codes) == [
        Path.MOVETO, Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    plt.close(backend.axis.figure)


def test_draw_curved_polygon_tikz():
    # TikZ implements the same generic draw_curved_polygon primitive as
    # Matplotlib, e.g. so that region drawing could be wired up for it too.
    backend = TikZ()
    backend.draw_curved_polygon(
        (0, 1), (1, 0), (2, 0), (2, 1), facecolor="red", bend_out=True)
    line = backend.edgelayer[-1]
    assert "controls" in line
    assert "fill={red}" in line


def test_draw_permutation():
    from matplotlib import pyplot as plt
    from discopy.monoidal import Box
    from discopy.symmetric import Ty, Permutation

    x, y, z = map(Ty, "xyz")
    perm = Permutation(x @ y @ z, [2, 0, 1])
    drawing = perm.to_drawing()
    box_node = drawing.box_nodes[0]
    assert len(list(drawing.graph.predecessors(box_node))) == len(perm.dom)
    assert len(list(drawing.graph.successors(box_node))) == len(perm.cod)
    assert drawing.box.draw_as_permutation == tuple(perm.perm)
    assert drawing.dagger().box.draw_as_permutation\
        == tuple(perm.perm.dagger())
    assert drawing.dagger() == perm.dagger().to_drawing()
    assert drawing.dagger().box.drawing_name\
        == perm.dagger().to_drawing().box.drawing_name

    swap = Permutation(x @ y, [1, 0]).to_drawing()
    swap.add_box_corners()
    tikz = TikZ()
    tikz.draw_wires(swap)
    assert len(tikz.edgelayer) == 2
    matplotlib = Matplotlib()
    matplotlib.draw_wires(swap)
    assert len(matplotlib.axis.patches) == 2
    plt.close(matplotlib.axis.figure)

    custom = Box(
        'custom', x @ y, y @ x, draw_as_wires=True,
        draw_as_permutation=(1, 0)).to_drawing()
    assert custom.dagger().dagger() == custom
    assert custom.dagger().dagger().box.name == 'custom'


def test_readable_foreground():
    # White and light colours get black text, dark colours get white text.
    assert Backend.readable_foreground("white") == "black"
    assert Backend.readable_foreground("black") == "white"
    assert Backend.readable_foreground("yellow") == "black"
    assert Backend.readable_foreground("darkblue") == "white"
    # Unrecognised colours fall back to black rather than raising.
    assert Backend.readable_foreground("not-a-colour") == "black"


def test_draw_box_foreground_on_dark_background():
    # A box with a dark custom colour gets a white label instead of the
    # default black, so its name stays legible.
    from matplotlib import pyplot as plt
    box = monoidal.Box(
        "f", monoidal.Ty("x"), monoidal.Ty("x"), color="black")
    drawing = box.to_drawing()
    drawing.add_box_corners()
    backend = Matplotlib(figsize=(2, 2))
    backend.draw_boxes(drawing)
    assert backend.axis.texts[-1].get_color() == "white"
    plt.close(backend.axis.figure)


def test_crack_two_eggs_at_once():
    from discopy.symmetric import Ty, Box, Diagram, Layer

    egg, white, yolk = Ty("egg"), Ty("white"), Ty("yolk")
    crack = Box("crack", egg, white @ yolk)
    merge = lambda X: Box("merge", X @ X, X)

    # DisCoPy allows string diagrams to be defined as Python functions

    @Diagram.from_callable(egg @ egg, white @ yolk)
    def crack_two_eggs(x, y):
        (a, b), (c, d) = crack(x), crack(y)
        return (merge(white)(a, c), merge(yolk)(b, d))

    # ... or in point-free style using parallel (@) and sequential (>>)
    # composition. from_callable returns the foliated diagram directly, so we
    # foliate the point-free staircase to compare.

    assert crack_two_eggs == (crack @ crack
        >> white @ Diagram.swap(yolk, white) @ yolk
        >> merge(white) @ merge(yolk)).foliation()

    assert crack_two_eggs.foliation() == Diagram(
        dom=egg @ egg, cod=white @ yolk, inside=(
            Layer(Ty(), crack, Ty(), crack, Ty()),
            Layer(white, Diagram.swap(yolk, white), yolk),
            Layer(Ty(), merge(white), Ty(), merge(yolk), Ty())))


@tikz_and_compare("spiral.tikz", wire_labels=False, use_tikzstyles=True)
def test_spiral_to_tikz():
    return spiral(2)


@tikz_and_compare("copy.tikz", use_tikzstyles=True)
def test_copy_to_tikz():
    x, y = map(Ty, ("$x$", "$y$"))
    copy_x, copy_y = Box('COPY', x, x @ x), Box('COPY', y, y @ y)
    copy_x.draw_as_spider, copy_y.draw_as_spider = True, True
    copy_x.drawing_name, copy_y.drawing_name = "", ""
    copy_x.color, copy_y.color = "black", "black"
    return copy_x @ copy_y >> Id(x) @ Swap(x, y) @ Id(y)


@tikz_and_compare("snake-equation.tikz", textpad=(.2, .2), textpad_words=(0, .25))
def test_snake_equation_to_tikz():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


@tikz_and_compare("who-ansatz.tikz")
def test_who_ansatz_to_tikz():
    from discopy.grammar.pregroup import Ty, Cap, Word, Id, Box
    s, n = Ty('s'), Ty('n')
    who = Word('who', n.r @ n @ s.l @ n)
    who_ansatz = Cap(n.r, n)\
        >> Id(n.r) @ Box('copy', n, n @ n)\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ Box('update', n @ s, n) @ Id(s.l @ n)
    return Equation(who, who_ansatz, symbol="$\\mapsto$")


@tikz_and_compare('bialgebra.tikz', use_tikzstyles=True)
def test_tikz_bialgebra_law():
    from discopy.quantum.zx import Z, X, Id, SWAP
    source = X(2, 1) >> Z(1, 2)
    target = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return Equation(source, target)


@tikz_and_compare('bell-state.tikz', aspect='equal', use_tikzstyles=True)
def test_tikz_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, CX
    H.draw_as_spider, H.color, H.drawing_name = True, "yellow", ""
    return sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit


@tikz_and_compare('crack-eggs.tikz')
def test_tikz_eggs():
    def merge(x):
        box = Box('merge', x @ x, x, draw_as_spider=True)
        return box

    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    return crack @ crack\
        >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
        >> merge(white) @ merge(yolk)


@tikz_and_compare('long-controlled.tikz', wire_labels=False)
def test_tikz_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (Controlled(CX.l, distance=3) >> Controlled(
        Controlled(CZ.l, distance=2), distance=-1))


def test_rich_display():
    from io import StringIO
    import matplotlib.pyplot as plt
    from discopy.monoidal import Ty, Box

    f = Box('f', Ty('x'), Ty('y'))
    diagram, drawing, equation = f, f.to_drawing(), Equation(f, f)
    plt.close('all')

    for obj in (diagram, drawing, equation):
        svg, png = obj._repr_svg_(), obj.to_png()
        assert svg.startswith('<?xml') and '</svg>\n' in svg
        assert png.startswith(b'\x89PNG')
        assert obj._repr_mimebundle_() == {
            'image/svg+xml': svg, 'image/png': png}
        assert obj._repr_mimebundle_(include=['image/svg+xml']) == {
            'image/svg+xml': svg}
        assert obj._repr_mimebundle_(exclude=['image/svg+xml']) == {
            'image/png': png}
        assert obj._repr_mimebundle_(include=['text/html']) == {}

    assert plt.get_fignums() == []

    # Output is deterministic, i.e. the same diagram renders byte-for-byte.
    assert diagram._repr_svg_() == diagram._repr_svg_()
    assert diagram.to_png() == diagram.to_png()

    # The format parameter of draw allows rendering to an in-memory buffer.
    buffer = StringIO()
    diagram.draw(path=buffer, format='svg', show=False)
    assert buffer.getvalue() == diagram.to_svg()

    import subprocess
    import sys
    script = """
from discopy.monoidal import Box, Ty
x = Ty("x")
boxes = [Box(str(i), x, x, draw_as_spider=True) for i in range(4)]
for box, color, shape in zip(
        boxes, ("red", "blue", "green", "yellow"),
        ("circle", "rectangle", "circle", "rectangle")):
    box.color, box.shape, box.drawing_name = color, shape, ""
print((boxes[0] @ boxes[1] @ boxes[2] @ boxes[3]).to_svg())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED=str(seed)))
        for seed in range(3)]
    assert outputs[0] == outputs[1] == outputs[2]


RIBBON_COLOURS = ("red", "green", "blue", "yellow")


def auto_colour_ribbons(diagram):
    """
    A colour map for nice looking dual rail examples: cycles through
    ``RIBBON_COLOURS`` assigning one colour per distinct object. An object
    and its adjoint encode the same wire, hence share the same colour.
    """
    obs = list(diagram.dom.inside)
    for box in diagram.boxes:
        obs += list(box.dom.inside) + list(box.cod.inside)
    palette = {name: RIBBON_COLOURS[i % len(RIBBON_COLOURS)]
               for i, name in enumerate(sorted({ob.name for ob in obs}))}
    return lambda ob: palette[ob.name]


def test_draw_ribbon_colors():
    # The inside of each ribbon is filled with a colour in the dual rail
    # drawing of a ribbon diagram, covering the straight rails, the cups, caps
    # and braids, with the colour and width preserved across the adjoint.
    from discopy.ribbon import Ty, Braid
    x = Ty('x')
    diagram = Braid(x, x).trace(left=False)
    diagram.to_ribbons(colour=auto_colour_ribbons(diagram)).draw(
        wire_labels=False, aspect='equal', show=False,
        doctest="docs/_static/ribbon/ribbon-colors.svg")


@tikz_and_compare('ribbon-colors.tikz', wire_labels=False)
def test_tikz_ribbon_colors():
    from discopy.ribbon import Ty, Braid
    x = Ty('x')
    diagram = Braid(x, x).trace(left=False)
    return diagram.to_ribbons(colour=auto_colour_ribbons(diagram))


def test_draw_twist_colors():
    # The back of a twisting ribbon, where it turns over, is filled with a
    # darker shade of the colour filling its front.
    from discopy.ribbon import Ty, Diagram
    x = Ty('x')
    Diagram.twist(x).to_ribbons().draw(
        wire_labels=False, aspect='equal', show=False,
        doctest="docs/_static/ribbon/twist-colors.svg")


@tikz_and_compare('twist-colors.tikz', wire_labels=False)
def test_tikz_twist_colors():
    from discopy.ribbon import Ty, Diagram
    x = Ty('x')
    return Diagram.twist(x).to_ribbons()


def test_darken():
    # A darker shade of a colour keeps each RGB channel smaller, and the
    # darker shades filling the back of a twisting ribbon are precomputed
    # for every named colour, see discopy.config.COLORS.
    from discopy.config import COLORS, darken
    for name in ["red", "green", "blue", "yellow", "gray"]:
        hexcode, dark_hexcode = COLORS[name], COLORS[f"dark_{name}"]
        assert darken(hexcode) == dark_hexcode
        channels = [int(hexcode[i:i + 2], 16) for i in (1, 3, 5)]
        dark_channels = [int(dark_hexcode[i:i + 2], 16) for i in (1, 3, 5)]
        assert all(d <= c for d, c in zip(dark_channels, channels))
        assert any(d < c for d, c in zip(dark_channels, channels))


def test_draw_nested_ribbons():
    # Nested cups and caps stay folds of constant (ribbon) width, i.e. the
    # inner ribbon is squeezed just like the outer one.
    from discopy.ribbon import Ty, Diagram
    x, y = Ty('x'), Ty('y')
    (Diagram.caps(x @ y, (x @ y).r)
        >> Diagram.cups(x @ y, (x @ y).r)).to_ribbons().draw(
        wire_labels=False, aspect='equal', show=False,
        doctest="docs/_static/ribbon/nested-ribbons.svg")
