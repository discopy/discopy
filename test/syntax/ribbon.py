from discopy.ribbon import *


def test_to_ribbons_width():
    x = Ty('x')

    # By default the two wires of each ribbon are four times closer.
    drawing = Diagram.twist(x).to_ribbons().to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    assert round(dom[1] - dom[0], 3) == 0.25

    # width=1 leaves the wires at the usual minimal width.
    drawing = Diagram.twist(x).to_ribbons(width=1).to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    assert round(dom[1] - dom[0], 3) == 1.0


def test_to_ribbons_trace_width():
    x = Ty('x')

    # The looped ribbon of a trace stays compressed all the way around, even
    # though its wires are rotated (which drops the per-object margin).
    drawing = Braid(x, x).trace(left=True).to_ribbons().to_drawing()
    rows = {}
    for node, point in drawing.positions.items():
        if node.kind in ("box_dom", "box_cod"):
            rows.setdefault(round(point.y, 3), []).append(round(point.x, 3))
    for xs in rows.values():
        xs = sorted(xs)
        gaps = [round(xs[i + 1] - xs[i], 3) for i in range(len(xs) - 1)]
        assert gaps == [0.25, 1.0, 0.25]


def test_to_ribbons_gadgets():
    x = Ty('x')
    trace = Braid(x, x).trace(left=True)

    # Cups, caps, braids and twists each become a single dual rail box.
    boxes = trace.to_ribbons(width=None).boxes
    assert any(isinstance(b, DualRailCap) for b in boxes)
    assert any(isinstance(b, DualRailCup) for b in boxes)
    assert any(isinstance(b, DualRailBraid) for b in boxes)

    cup, = (b for b in boxes if isinstance(b, DualRailCup))
    assert len(cup.dom) == 4 and len(cup.cod) == 0
    assert isinstance(cup.dagger(), DualRailCap)


def test_Kauffman():
    tmp = Ty.l, Ty.r
    Ty.l = Ty.r = property(lambda self: self)
    x, A = Ty('x'), Box('A', Ty(), Ty())

    class Polynomial(Diagram):
        def braid(x, y):
            return (A @ x @ y) + (Cup(x, y) >> A.dagger() >> Cap(x, y))

    Kauffman = Functor(ob={x: x}, ar={}, cod=Polynomial)

    assert Kauffman(Braid(x, x))\
        == (A @ x @ x) + (Cup(x, x) >> A.dagger() >> Cap(x, x))

    Ty.l, Ty.r = tmp


def test_rotate():
    x = Ty('x')
    assert Twist(x).r == Twist(x)
