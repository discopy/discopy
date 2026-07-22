
import os

from pytest import raises
from matplotlib.testing.compare import compare_images

from discopy.utils import AxiomError
from discopy.config import DRAWING_DEFAULT
from discopy.compact import *
from discopy.drawing import *
from discopy import monoidal

IMG_FOLDER, TIKZ_FOLDER, TOL = 'test/drawing/imgs/', 'test/drawing/tikz/', 20


def draw_and_compare(file, folder=IMG_FOLDER, **params):
    tol = params.pop('tol', TOL)

    def decorator(func):
        def wrapper():
            diagram = func()
            draw = params.get('draw', type(diagram).draw)
            true_path = os.path.join(folder, file)
            test_path = os.path.join(folder, '_' + file)
            draw(diagram, path=test_path, show=False, **params)
            if not os.path.exists(true_path):
                os.replace(test_path, true_path)
                return
            test = compare_images(true_path, test_path, tol)
            assert test is None
            os.remove(test_path)
        return wrapper
    return decorator


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


@draw_and_compare('crack-eggs.png')
def test_draw_eggs():
    def merge(x):
        return Box('merge', x @ x, x)

    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    return crack @ crack\
        >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
        >> merge(white) @ merge(yolk)


def test_draw_coloured_regions_and_frame():
    red, green, blue = map(
        monoidal.Colour, ("red", "green", "blue"))
    x = monoidal.Ty(monoidal.Wire("x", red, green))
    y = monoidal.Ty(monoidal.Wire("y", green, blue))
    z = monoidal.Ty(monoidal.Wire("z", red, blue))
    box = monoidal.Box("f", x @ y, z)
    outer = monoidal.Ty(monoidal.Wire("u", blue, red))
    # A box fills its three wire regions.
    assert {'#ff0000', '#008000', '#0000ff'} <= region_hexes(box)
    # A frame additionally fills its frame background (lightgrey).
    frame = box.bubble(dom=outer, cod=outer, draw_as_frame=True)
    assert {'#ff0000', '#008000', '#0000ff', '#d3d3d3'} <= region_hexes(frame)


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


# A higher tolerance: abutting high-contrast regions turn a sub-pixel
# boundary shift across environments into a large RMS at tol=20.
@draw_and_compare('coloured-bubble.png', wire_labels=False, tol=50)
def test_draw_bubble():
    return coloured_bubble()


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


# A higher tolerance: abutting high-contrast regions turn a sub-pixel
# boundary shift across environments into a large RMS at tol=20.
@draw_and_compare('coloured-frame.png', wire_labels=False, tol=50)
def test_draw_coloured_frame():
    red, blue = map(monoidal.Colour, ("red", "blue"))
    x = monoidal.Ty(monoidal.Wire("x", red, blue))
    boundary = monoidal.Ty(monoidal.Wire("boundary", blue, red))
    return monoidal.Box("f", x, x).bubble(
        dom=boundary, cod=boundary, draw_as_frame=True)


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
    # A cup and a cap each separate the two boundary regions.
    assert region_hexes(Cup(x, x.r)) == {'#ff0000', '#008000'}
    assert region_hexes(Cap(x.r, x)) == {'#ff0000', '#008000'}


def test_draw_coloured_crossings_are_monochrome():
    from discopy.frobenius import Spider, Ty as FTy, Ob as FOb
    red = monoidal.Colour("red")
    # Wires that cross or merge must be globular, i.e. carry the same colour
    # on both sides, so their regions are a single colour.
    assert region_hexes(Swap(
        Ty(Ob("x", dom=red, cod=red)), Ty(Ob("y", dom=red, cod=red)))
    ) == {'#ff0000'}
    assert region_hexes(Spider(2, 1, FTy(FOb("x", dom=red, cod=red)))) == {
        '#ff0000'}
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
    assert {'#ff0000', '#008000', '#ffffff'} <= colours


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
    assert swatches == {'#ff0000', '#008000', '#0000ff'}
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


@draw_and_compare('crack-two-eggs-at-once.png')
def test_crack_two_eggs_at_once():
    from discopy.monoidal import Layer
    from discopy.symmetric import Ty, Box, Diagram

    egg, white, yolk = Ty("egg"), Ty("white"), Ty("yolk")
    crack = Box("crack", egg, white @ yolk)
    merge = lambda X: Box("merge", X @ X, X)

    # DisCoPy allows string diagrams to be defined as Python functions

    @Diagram.from_callable(egg @ egg, white @ yolk)
    def crack_two_eggs(x, y):
        (a, b), (c, d) = crack(x), crack(y)
        return (merge(white)(a, c), merge(yolk)(b, d))

    # ... or in point-free style using parallel (@) and sequential (>>) composition

    assert crack_two_eggs == crack @ crack\
        >> white @ Diagram.swap(yolk, white) @ yolk\
        >> merge(white) @ merge(yolk)

    crack_two_eggs_at_once = crack_two_eggs.foliation()

    assert crack_two_eggs_at_once == Diagram(
        dom=egg @ egg, cod=white @ yolk, inside=(
            Layer(Ty(), crack, Ty(), crack, Ty()),
            Layer(white, Diagram.swap(yolk, white), yolk),
            Layer(Ty(), merge(white), Ty(), merge(yolk), Ty())))

    return crack_two_eggs_at_once


@draw_and_compare("bubble-straight-wire.png", wire_labels=False)
def test_draw_bubble_wires():
    return (Ty('x') @ Box('s', Ty(), Ty())).bubble()


@draw_and_compare(
    'spiral.png', wire_labels=False,
    draw_box_labels=False, aspect='equal')
def test_draw_spiral():
    return spiral(2)


@draw_and_compare('who-ansatz.png', aspect='equal')
def test_draw_who():
    n, s = Ty('n'), Ty('s')
    copy, update = Box('copy', n, n @ n), Box('update', n @ s, s)
    return Cap(n.r, n)\
        >> Id(n.r) @ copy\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ update @ Id(s.l @ n)


@draw_and_compare('alice-loves-bob.png')
def test_draw_pregroup_sentence():
    from discopy.grammar.pregroup import Ty, Cup, Word, Id
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
    return sentence.foliation()


@draw_and_compare('categorial-grammar.png', aspect='equal')
def test_draw_sentence():
    from discopy.grammar.categorial import Eval, Ty, Word

    s, n = map(Ty, 'sn')

    Alice = Word('Alice', n)
    loves = Word('loves', (n >> s) << n)
    Bob = Word('Bob', n)

    return Alice @ loves @ Bob >> n @ Eval((n >> s) << n) >> Eval(n >> s)


@draw_and_compare('bialgebra.png', aspect='equal')
def test_draw_bialgebra():
    from discopy.quantum.zx import Z, X, Id, SWAP
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return bialgebra + bialgebra


@draw_and_compare("snake-equation.png",
                  aspect='auto', figsize=(5, 2), wire_labels=False)
def test_snake_equation():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


@draw_and_compare('typed-snake-equation.png', figsize=(4, 1))
def test_draw_typed_snake():
    from discopy.rigid import Ty, Id
    x = Ty('x')
    return Equation(Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose())


# tol=50: the abutting coloured regions make this sensitive to sub-pixel
# boundary shifts across environments, as for test_draw_coloured_frame.
@draw_and_compare(
    'coloured-snake-equation.png', figsize=(3, 2), legend=True, tol=50)
def test_draw_coloured_snake_equation():
    from discopy.rigid import Ty, Ob, Id, Cup, Cap
    a = monoidal.Colour("cornflowerblue", label="Function")
    b = monoidal.Colour("palegreen", label="Morphism")
    F = Ty(Ob("F", dom=a, cod=b))
    G = F.r
    return Equation(Id(F) @ Cap(G, F) >> Cup(F, G) @ Id(F), Id(F))


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


@draw_and_compare('empty_diagram.png')
def test_empty_diagram():
    return Id()


@draw_and_compare('bell-state.png', aspect='equal')
def test_draw_bell_state():
    from discopy.quantum import qubit, H, sqrt, Bra, Ket, CX
    return sqrt(2) >> Ket(0, 0) >> H @ qubit >> CX >> Bra(0) @ qubit


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


@draw_and_compare('long-controlled.png', wire_labels=False, tol=5)
def test_draw_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (Controlled(CX.l, distance=3) >> Controlled(
        Controlled(CZ.l, distance=2), distance=-1))


@tikz_and_compare('long-controlled.tikz', wire_labels=False)
def test_tikz_long_controlled():
    from discopy.quantum import Controlled, CZ, CX
    return (Controlled(CX.l, distance=3) >> Controlled(
        Controlled(CZ.l, distance=2), distance=-1))


def classical_controlled():
    # A controlled gate over distinct wires, e.g. a classically-controlled
    # gate, used to hit a KeyError looking up its nodes with the wrong type.
    bit, qubit = monoidal.Ty("bit"), monoidal.Ty("qubit")
    gate = monoidal.Box("F", qubit, qubit)
    controlled = monoidal.Box(
        "CF", bit @ qubit, bit @ qubit,
        draw_as_controlled=True, controlled=gate, distance=1)
    left_controlled = monoidal.Box(
        "FC", qubit @ bit, qubit @ bit,
        draw_as_controlled=True, controlled=gate, distance=-1)
    return controlled @ left_controlled


@draw_and_compare('controlled-classical.png')
def test_draw_controlled_classical():
    return classical_controlled()


@tikz_and_compare('controlled-classical.tikz')
def test_tikz_controlled_classical():
    return classical_controlled()


def test_tikz_controlled_node_ids():
    # Nested controlled gates put several nodes at the same point, which used
    # to make TikZ output duplicate node ids and misdirected control wires.
    import re
    from discopy.quantum import Controlled, X
    path = os.path.join(TIKZ_FOLDER, '_ccx-node-ids.tikz')
    Controlled(Controlled(X)).draw(path=path, to_tikz=True)
    with open(path, "r") as file:
        lines = file.read().splitlines()
    os.remove(path)
    node_ids = [re.search(r"\((\d+)\) at", line).group(1)
                for line in lines if line.startswith("\\node ")]
    assert len(node_ids) == len(set(node_ids))
    wires = [re.findall(r"\((\d+)\.center\)", line)
             for line in lines if "out=" in line]
    assert all(source != target for source, target in wires)


@draw_and_compare('long-box-name.png', aspect='equal')
def test_draw_long_box_name():
    # A box gets wider when its name is too long to fit between its wires,
    # while boxes with short names keep their default size.
    x = Ty('x')
    return Box('f', x, x @ x)\
        >> Box('a_box_with_a_very_long_name', x @ x, x)\
        >> Box('g', x, x)


@draw_and_compare('box-min-width.png', aspect='equal')
def test_draw_box_min_width():
    # The width of a box can be set by hand with `min_width`, e.g. for a
    # LaTeX name whose rendered width cannot be guessed from its characters.
    x = Ty('x')
    return Box('$\\Lambda$', x, x, min_width=3) @ Box('f', x, x)


@draw_and_compare('wire-min-right-margin.png', aspect='equal')
def test_draw_wire_min_right_margin():
    # An object's `min_right_margin` adds space to the right of its wire,
    # e.g. to fit a long label without colliding with the next wire.
    x, long_type = Ty('x'), Ty('a_long_type_name')
    long_type.inside[0].min_right_margin = 1.5
    return Id(x @ long_type @ x)


@draw_and_compare('wire-custom-margin.png', aspect='equal')
def test_draw_wire_custom_margin():
    x, custom = Ty('x'), Ty('custom_margin_wire')
    custom.inside[0].right_margin = 3
    return Id(x @ custom @ x)


@draw_and_compare('wire-auto-margin.png', aspect='equal')
def test_draw_wire_auto_margin():
    # A long wire label reserves space to its right on its own, so it does
    # not overflow even without setting min_right_margin by hand.
    x = Ty('x')
    return Box('f', x, x @ Ty('a_long_output_type'))


@draw_and_compare('long-latex-name.png', aspect='equal', tol=100)
def test_draw_long_latex_name():
    x = Ty('x')
    return Box('$\\int_a^b f(x)\\,dx = \\sqrt{2}$', x, x) @ Box('f', x, x)


def test_to_gif():
    from discopy.grammar.pregroup import (
         Ty, Cup, Cap, Box, Word, Functor)

    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)

    sentence = Alice @ loves @ Bob\
        >> Cup(n, n.r) @ s @ Cup(n.l, n)

    def wiring(word):
        if word.cod == n:  # word is a noun
            return word
        if word.cod == n.r @ s @ n.l:  # word is a transitive verb
            box = Box(word.name, n @ n, s)
            return Cap(n.r, n) @ Cap(n, n.l) >> n.r @ box @ n.l

    W = Functor(ob_map={s: s, n: n}, ar_map=wiring)

    rewrite_steps = W(sentence).normalize()
    params = dict(
        path=IMG_FOLDER + 'autonomisation.gif', timestep=1000, figsize=(4, 4))
    sentence.to_gif(*rewrite_steps, **params)
