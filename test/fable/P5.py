"""P5: substitution invariance, subs and lambdify only touch the data.
Miniature of the property over curated examples; red while its bullets (B11) are live — issue #606.
"""
import sympy

from discopy.quantum.gates import CRz, scalar


def test_p5():
    phi = sympy.Symbol('phi')
    examples = [
        ("CRz(phi, distance=2)", CRz(phi, distance=2), "distance", 2),
        ("scalar(2*phi, is_mixed=True)",
         scalar(2 * phi, is_mixed=True), "is_mixed", True),
    ]
    failures = []
    for label, box, attr, expected in examples:
        try:
            found = getattr(box.subs(phi, .5), attr, None)
            if found != expected:
                failures.append(
                    f"{label}: subs(phi, .5) turned {attr}={expected}"
                    f" into {attr}={found}")
        except Exception as error:
            failures.append(f"{label}: subs raised {type(error).__name__}")
        try:
            found = getattr(box.lambdify(phi)(.5), attr, None)
            if found != expected:
                failures.append(
                    f"{label}: lambdify(phi)(.5) turned {attr}={expected}"
                    f" into {attr}={found}")
        except Exception as error:
            failures.append(f"{label}: lambdify raised {type(error).__name__}")
    assert not failures, failures
