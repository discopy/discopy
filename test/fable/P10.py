"""P10: foliation is an equality — d.foliation() is d up to interchangers and depth() never raises.
Miniature of the property over curated examples; red while its bullets (B49, B61, B73) are live — issue #699.
"""
from discopy import monoidal, symmetric, traced, rigid


def _planar(d):
    return d.to_staircases().normal_form()


def _hypergraph(d):
    return d.to_hypergraph()


def _cases():
    x, y = monoidal.Ty('x'), monoidal.Ty('y')
    f, g, h = monoidal.Box('f', x, y), monoidal.Box('g', y, x), monoidal.Box('h', x, x)
    sx, sy = symmetric.Ty('x'), symmetric.Ty('y')
    sf, sg, sh = (symmetric.Box('f', sx @ sx, sx @ sx),
                  symmetric.Box('g', sy @ sx @ sx, sx), symmetric.Box('h', sx, sx))
    tx = traced.Ty('x')
    tf, th = traced.Box('f', tx @ tx, tx @ tx), traced.Box('h', tx, tx)
    rx = rigid.Ty('x')
    snake = rx @ rigid.Cap(rx.r, rx) >> rigid.Cup(rx, rx.r) @ rx
    return [
        ("monoidal, two layers of two boxes (control)", f @ g >> g @ f, _planar),
        ("monoidal, named bubble beside a box", f.bubble(name="N") @ h, _planar),
        ("symmetric, permutation then box (control)",
         symmetric.Permutation(sx @ sy @ sx, [1, 2, 0]) >> sg, _hypergraph),
        ("symmetric, trace beside a box", sf.trace() @ sh, _hypergraph),
        ("traced, trace beside a box", tf.trace() @ th, _hypergraph),
        ("rigid, snake alone", snake, _hypergraph),
        ("rigid, box beside a snake", rigid.Box('g', rx, rx) @ snake, _hypergraph),
    ]


def test_p10():
    failures = []
    for label, d, up_to in _cases():
        try:
            fol = d.foliation()
        except Exception as error:
            failures.append(f"{label}: foliation raised {type(error).__name__}")
            continue
        if (fol.dom, fol.cod) != (d.dom, d.cod):
            failures.append(f"{label}: foliation changed dom or cod")
        if sorted(map(repr, fol.boxes)) != sorted(map(repr, d.boxes)):
            failures.append(f"{label}: foliation changed the boxes")
        try:
            if up_to(fol) != up_to(d):
                failures.append(f"{label}: foliation is not equal up to interchangers")
        except Exception as error:
            failures.append(f"{label}: comparison raised {type(error).__name__}")
        try:
            d.depth()
        except Exception as error:
            failures.append(f"{label}: depth raised {type(error).__name__}")
    assert not failures, failures
