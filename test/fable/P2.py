"""P2: dagger and rotation laws, dagger is involutive and r, l are inverse.
Miniature of the property over curated examples; red while its bullets (B17, B18, B21) are live — issue #606.
"""
from discopy import cat, monoidal, rigid, ribbon


def test_p2():
    x, y = rigid.Ty('x'), rigid.Ty('y')
    rx, ry = ribbon.Ty('x'), ribbon.Ty('y')
    mx, my = monoidal.Ty('x'), monoidal.Ty('y')
    examples = [
        ("rigid rotated box f.r", rigid.Box('f', x, y).r),
        ("ribbon Braid.dagger()", ribbon.Braid(rx, ry).dagger()),
        ("cat bubble", cat.Box('f', cat.Ob('x'), cat.Ob('y')).bubble()),
        ("monoidal box", monoidal.Box('f', mx, my)),
    ]
    failures = []
    for label, b in examples:
        try:
            if b[::-1][::-1] != b:
                failures.append(f"{label}: b[::-1][::-1] != b")
        except Exception as error:
            failures.append(
                f"{label}: b[::-1][::-1] raised {type(error).__name__}")
        if not hasattr(type(b), "r"):
            continue
        try:
            if b.r.l != b:
                failures.append(f"{label}: b.r.l != b")
            if (b.r.dom, b.r.cod) != (b.cod.r, b.dom.r):
                failures.append(
                    f"{label}: b.r has dom {b.r.dom} -> cod {b.r.cod}, "
                    f"expected {b.cod.r} -> {b.dom.r}")
        except Exception as error:
            failures.append(f"{label}: rotation raised {type(error).__name__}")
    assert not failures, failures
