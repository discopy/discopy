from discopy.balanced import *


def test_repr():
    x = Ty('x')
    assert repr(Twist(Ty('x')))\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x')))"
    assert repr(Twist(Ty('x')).dagger())\
        == "balanced.Twist(monoidal.Ty(cat.Ob('x'))).dagger()"


def test_double_rail():
    x = Ty('x')
    # Doubling sets the first rail's margin so the two rails are `width` apart.
    rail = double_rail(x, .25)
    assert len(rail) == 2
    margins = [getattr(o, "min_right_margin", 0) for o in rail.inside]
    assert margins == [-0.75, 0]


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


def test_to_braided_default_and_zero_width():
    from discopy import config

    x = Ty('x')
    twist = Diagram.twist(x)

    # width=None pulls the default width from discopy.config.
    assert twist.to_braided() == twist.to_braided(config.RIBBON_WIDTH)

    # width=0 returns the diagram as is, i.e. without dual rails.
    assert twist.to_braided(width=0) == twist
