"""P6: the identity functor is the identity on every box.
Miniature of the property over curated examples; red while its bullets (B16) are live — issue #606.
"""
from discopy import frobenius


def test_p6():
    x = frobenius.Ty('x')
    functor = frobenius.Functor(ob_map=lambda ob: ob, ar_map=lambda ar: ar)
    examples = [
        ("plain Box", frobenius.Box('f', x, x)),
        ("Spider(1, 2, x)", frobenius.Spider(1, 2, x)),
        ("Spider(1, 1, x, phase=0.5)", frobenius.Spider(1, 1, x, 0.5)),
    ]
    failures = []
    for label, box in examples:
        try:
            if functor(box) != box:
                failures.append(f"{label}: F(b) != b for the identity functor")
        except Exception as error:
            failures.append(f"{label}: F(b) raised {type(error).__name__}")
    assert not failures, failures
