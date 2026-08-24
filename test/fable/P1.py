"""P1: transparency round-trip, eval(repr(x)) == x and loads(dumps(x)) == x.
Miniature of the property over curated examples; red while its bullets (B16, B21, B39) are live — issue #606.
"""
from discopy import (cat, monoidal, traced, frobenius, rigid, pivotal,
                     compact, quantum, ribbon, interaction, grammar, utils)
from discopy.quantum import zx
from discopy.grammar.pregroup import Word

NS = dict(cat=cat, monoidal=monoidal, traced=traced, frobenius=frobenius,
          rigid=rigid, pivotal=pivotal, compact=compact, quantum=quantum,
          ribbon=ribbon, interaction=interaction, grammar=grammar,
          zx=zx, Word=Word)


def test_p1():
    x, y = monoidal.Ty('x'), monoidal.Ty('y')
    tx, fx = traced.Ty('x'), frobenius.Ty('x')
    cx, cy, cz = map(compact.Ty, "xyz")
    x0, x1, y0, y1 = map(ribbon.Ty, ["x0", "x1", "y0", "y1"])
    X = interaction.Ty[ribbon.Ty](x0, x1)
    Y = interaction.Ty[ribbon.Ty](y0, y1)
    examples = [
        ("monoidal.Box", monoidal.Box('f', x, y)),
        ("traced.Trace(left=False)",
         traced.Box('f', tx @ tx, tx @ tx).trace(left=False)),
        ("cat.Bubble(name=, method=)",
         cat.Box('f', cat.Ob('x'), cat.Ob('y')).bubble(name='n', method='m')),
        ("frobenius.Spider(phase)", frobenius.Spider(1, 2, fx, 0.5)),
        ("quantum.Measure()", quantum.Measure()),
        ("zx.scalar(2)", zx.scalar(2)),
        ("compact.CMap",
         compact.CMap.from_box(compact.Box('f', cx @ cy, cx @ cz))),
        ("interaction.Diagram", interaction.Diagram[ribbon.Diagram](
            ribbon.Box('f', x0 @ y1, y0 @ x1), X, Y)),
        ("pregroup.Word", Word('Alice', grammar.pregroup.Ty('n'))),
    ]
    failures = []
    for label, b in examples:
        try:
            if eval(repr(b), dict(NS)) != b:
                failures.append(f"{label}: eval(repr(x)) != x")
        except Exception as error:
            failures.append(
                f"{label}: eval(repr(x)) raised {type(error).__name__}")
        try:
            if utils.loads(utils.dumps(b)) != b:
                failures.append(f"{label}: loads(dumps(x)) != x")
        except Exception as error:
            failures.append(
                f"{label}: loads(dumps(x)) raised {type(error).__name__}")
    assert not failures, failures
