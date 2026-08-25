from discopy.balanced import *


def test_repr():
    x = Ty('x')
    assert repr(Twist(Ty('x')))\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x')))"
    assert repr(Twist(Ty('x')).dagger())\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x'))).dagger()"


def test_double_rail():
    x = Ty('x')
    # Doubling puts a shared Ribbon region between the two rails of a wire,
    # gray by default, carrying both the colour and the width of the ribbon.
    rail = double_rail(x, .25)
    left, right = rail.inside
    assert len(rail) == 2
    assert left.cod is right.dom == Ribbon("gray", width=.25)


def test_double_rail_colour():
    x = Ty('x')
    rail = double_rail(x, .25, colour="red")
    assert rail.inside[0].cod == Ribbon("red", width=.25)


def test_double_rail_colour_callable():
    x = Ty('x')
    rail = double_rail(x, colour=lambda ob: ob.name.upper())
    assert rail.inside[0].cod.name == "X"


def test_ribbon_survives_adjoint():
    from discopy.pivotal import Ty as PivotalTy
    x = PivotalTy('x')
    # The adjoint reverses the two rails and swaps the sides of each object,
    # so the pair stays one colour region of the same colour and width.
    rail = double_rail(x, .25, colour="red")
    ribbon = rail.inside[0].cod
    adjoint = rail.r
    assert adjoint.inside[0].cod is ribbon is adjoint.inside[1].dom


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
    assert braid.dom == double_rail(x @ y) and braid.cod == double_rail(y @ x)
    assert braid.dagger().dom == braid.cod and braid.dagger().cod == braid.dom

    twist, = Diagram.twist(x).to_braided(width=None).boxes
    assert isinstance(twist, DualRailTwist)
    assert twist.dom == twist.cod == double_rail(x)


def test_to_braided_default_and_zero_width():
    from discopy import config

    x = Ty('x')
    twist = Diagram.twist(x)

    # width=None pulls the default width from discopy.config.
    assert twist.to_braided() == twist.to_braided(
        config.DRAWING_DEFAULT["ribbon_width"])

    # width=0 returns the diagram as is, i.e. without dual rails.
    assert twist.to_braided(width=0) == twist
