# TODO

> File an issue in discopy about making Quimb a tensor backend. Check the best way to include quimb in the backends, moving the method to_quimb from tensor.Diagram to tensor.CMap, reusing CMap.from_diagram, it seems to me that it should be more efficient.

> CMap.from_diagram seems excessively slow, I presume Diagram.to_cmap is the same if it exists. That's a problem. I want the most efficient route to evaluation of tensor.Diagrams, quimb but also the other backends. Are they also using this slow from_diagram?

> Actually file an issue about from_diagram and open a PR with the fix

- [x] File issue #523 on making quimb a tensor backend via `tensor.CMap`
- [x] File issue #524 on `CMap.from_diagram` being quadratic
- [WIP] @013791a6-2026-08-04 12:00 Make `CMap.from_diagram` a single pass: `CMap.from_glued` glues box images
      onto a scan of open wires, `validate` runs once on the finished map
- [ ] Stop `CMap.__init__` materialising every `Port` just to count them
- [ ] Tests: `from_glued` agrees with the fold, cups/caps/loops/multi-box layers
- [ ] `CHANGELOG.md` entry under `[Unreleased]` / Performance
- [ ] `pflake8 discopy` and `coverage run -m pytest` green
