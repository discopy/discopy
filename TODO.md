> toumix, 2026-10-26, on `monoidal.is_monochrome`: "Code smell: this None shouldn't be here, this whole method shouldn't exist" and "That's not even what monochrome means ffs"
>
> cubic, 2026-10-26, requested by toumix: the generator round-trip loop is copied into 14 test modules; `test/traced.py` derives its expected repr from `repr(f)`; `decompositions` nests four loops deep; `Spider.strategy` never draws a phase; constrained layers only draw free boxes.

- [x] `is_monochrome` goes, by merging #817 now that #830 landed; toumix asked "Could we remove `transparent` and use `None` instead?", answered yes on https://github.com/discopy/discopy/pull/817#discussion_r4375687378, as its own PR from `main` that this one then merges
- [x] One helper for the generator round-trips, beside `assert_axioms`
- [x] `test_trace_repr` pins its literal
- [x] `decompositions` within three levels
- [x] `Spider.strategy` draws a phase
- [x] `Spider.__repr__` prints its phase as the argument `__init__` reads, where it printed `phase=` and did not read back: found by drawing phases
- [x] Constrained layers drawing free boxes only stays: proposal 2 of the description, answered on its thread
- [x] `assert_generators` draws single-shot with no example database, and checks each generator by its own `serialisation` and `transparency` laws (cubic's second pass)

> toumix, 2026-10-26, on #818: "All of these should go in axioms, they are not abstract base classes", "Seems like this should go in axioms rather than abc, same for TraceSliding below", "Again, all of these Testable subclasses should go to axioms", "Especially because while the machinery for axioms are defined there, the axioms themselves go in the abc module", "This makes me think the axioms module could be called `testable` instead" and "There should also be two other reidemeister 1 for left and right trace"

- [x] The shapes of the laws' arguments move from `abc` to `axioms`, the laws stay in `abc`
- [x] Reidemeister move 1 for the left and the right trace on `CompactCategory`
- [ ] `axioms` renamed `testable`: #829, from `main` with no alias (toumix: "no need for an alias, this axiom module was never released"), which this merges once it lands

> cubic, 2026-10-26, on #818: `compact.Diagram` restates currying though it has no `curry`; the free trace's inapplicable naturality reaches `pivotal`; `feedback.Trace` has no `delay`; right dinaturality slices `[-0:]`; a test catches `Exception`; `#562` is cited for currying; `CMap.is_boundary_connected` calls a rigid hypergraph partial.

- [x] `pivotal.Diagram` declares the four trace naturality laws broken, which they are there, rather than inheriting the free trace's inapplicable
- [x] `feedback.Trace.delay` delays what it traces
- [x] No `[-0:]` slice in the trace and feedback shapes
- [x] The shape test catches `AxiomError`, and the currying reason does not cite #562
- [x] Compact currying stays: `compact.Diagram` curries through cups and caps, its cells pass
- [x] The rigid hypergraph is partial: `rigid.Cap(x.l, x.l.l).to_hypergraph()` raises, the docstring stays
