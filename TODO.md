# Prompt (verbatim)

Fix discopy issue #623: "`Hypergraph.from_diagram` is quadratic: `then` relabels the whole hypergraph at every layer" (https://github.com/discopy/discopy/issues/623).

Full issue text:

---
`Diagram.to_hypergraph()` goes through `Hypergraph.functor`, which folds the layers with `Hypergraph.then`; `then` (`hypergraph.py:354`) computes the pushout and relabels *every* spider and box of both operands, then `__init__` relabels again by first occurrence. Each layer therefore costs the size of the hypergraph built so far.

Measured on main, a chain of `n` boxes with a swap after each (`Id(x@x) >> (Box @ x) >> Swap(x, x)` repeated):

| n | `Hypergraph.from_diagram` | `CMap.from_diagram` |
|---|---|---|
| 50 | 80 ms | 7 ms |
| 200 | 960 ms | 25 ms |
| 800 | 15.3 s | 103 ms |
| 3200 | — | 480 ms |

`CMap.from_diagram` is linear because #525 gave it `from_glued`, a single pass gluing each box onto a scan of open wires with a union-find. `Hypergraph` needs the same single pass (it is also what #449 asks for: `Hypergraph.from_diagram` should factor through `CMap`, or share the scan). `symmetric.Equation`, `compact.Equation`, `frobenius.Equation`, `simplify` and `foliation` all pay this.
---

## Plan

- [x] Reproduce the quadratic behaviour locally (informal timing script) to confirm the bug and later confirm the fix.
- [x] Read `CMap.from_glued`/`CMap.from_diagram` (#525) to understand the single-pass union-find gluing pattern.
- [x] Implement `Hypergraph.from_glued`, a single-pass union-find gluing of a sequence of hypergraphs onto a scan of open wires, mirroring `CMap.from_glued`.
- [x] Rewrite `Hypergraph.from_diagram` to flatten `old.inside` layers via `layer.boxes_and_offsets` and glue box images with `from_glued`, instead of folding with the generic `Functor`/`then`.
- [x] Add tests: a `from_glued` correctness example/doctest, a regression test that would catch a wrong-wiring optimisation (not just speed), and keep existing `test/hypergraph.py`, `test/symmetric.py`, `test/compact.py`, `test/frobenius.py` green.
- [x] Add a `### Performance` entry to `CHANGELOG.md`'s `[Unreleased]` section, linking #623.
- [x] Run `uv run pflake8 discopy` and `uv run coverage run -m pytest` (or `--skip-extra`) until clean and green.
- [x] Re-measure timings after the fix and record them.
