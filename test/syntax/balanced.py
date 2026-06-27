from discopy.balanced import *


def test_repr():
    x = Ty('x')
    assert repr(Twist(Ty('x')))\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x')))"
    assert repr(Twist(Ty('x')).dagger())\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x'))).dagger()"


def test_double_rail():
    x = Ty('x')
    # Doubling makes the two rails share a ribbon of the given width and colour.
    rail = double_rail(x, .25, color="red")
    assert len(rail) == 2
    left, right = rail.inside
    assert left.ribbon is right.ribbon
    assert left.ribbon.width == .25 and left.ribbon.color == "red"


def test_double_rail_color_callable():
    x = Ty('x')
    rail = double_rail(x, color=lambda ob: ob.name.upper())
    assert rail.inside[0].ribbon.color == "X"


def test_ribbon_survives_adjoint():
    from discopy.pivotal import Ty as PivotalTy
    x = PivotalTy('x')
    # Taking the adjoint reverses the two rails but keeps them a colour region.
    rail = double_rail(x, .25, color="red")
    ribbon = rail.inside[0].ribbon
    adjoint = rail.r
    assert [o.ribbon for o in adjoint.inside] == [ribbon, ribbon]


def test_to_braided_width():
    x, y = Ty('x'), Ty('y')

    # By default the two wires of each ribbon are four times closer.
    drawing = Diagram.twist(x @ y).to_braided().to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    gaps = [round(dom[i + 1] - dom[i], 3) for i in range(len(dom) - 1)]
    assert gaps == [0.25, 1.0, 0.25]

    # width=1 leaves the wires at the usual minimal width.
    drawing = Diagram.twist(x @ y).to_braided(width=1).to_drawing()
    dom = sorted(drawing.positions[n].x for n in drawing.dom_nodes)
    gaps = [round(dom[i + 1] - dom[i], 3) for i in range(len(dom) - 1)]
    assert gaps == [1.0, 1.0, 1.0]


def test_dual_rail_braid_and_twist():
    x, y = Ty('x'), Ty('y')

    # A swap of two ribbons becomes a single DualRailBraid, not four braids,
    # and a twist becomes a single DualRailTwist.
    braid, = Braid(x, y).to_braided(width=None).boxes
    assert isinstance(braid, DualRailBraid)
    assert braid.dom == x @ x @ y @ y and braid.cod == y @ y @ x @ x
    assert braid.dagger().dom == braid.cod and braid.dagger().cod == braid.dom

    twist, = Diagram.twist(x).to_braided(width=None).boxes
    assert isinstance(twist, DualRailTwist)
    assert twist.dom == twist.cod == x @ x
