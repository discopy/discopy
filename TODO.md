# TODO

> shapes in abc sounds good let's go with that

(USER, 2026-10-17, on the first proposal of #817, the clean first PR of the property-testing
re-cut asked for on 2026-10-16: *"Help daydream and I merge this property-based testing framework
starting with a clean PR to refactor the architecture of DisCoPy's modules with the minimum amount
of boilerplate, there must be a way to get there without a import cycle"*.)

- [x] the laws of every doctrine, stated once on their abstract base class in `abc.py` (from #659)
- [x] the argument shapes those laws quantify over, defined in `abc.py` beside the class whose
      law needs them, importing nothing but `axioms` (closes #711)
- [x] `Diagram.equation_factory` wired at every level, so a law compares up to the quotient of its
      category: strictly below `symmetric`, up to hypergraph from it on
- [ ] run the matrix; declare each law a free category breaks or has no structure for, with its
      reason, where the diagram class is — and only where the measurement says so
- [ ] tests for the shapes; `CHANGELOG.md`; `pflake8`, `pylint` under the gate, `pytest`, the matrix
