> toumix, 2026-10-26, on `monoidal.is_monochrome`: "Code smell: this None shouldn't be here, this whole method shouldn't exist" and "That's not even what monochrome means ffs"
>
> cubic, 2026-10-26, requested by toumix: the generator round-trip loop is copied into 14 test modules; `test/traced.py` derives its expected repr from `repr(f)`; `decompositions` nests four loops deep; `Spider.strategy` never draws a phase; constrained layers only draw free boxes.

- [ ] `is_monochrome` goes, waiting on toumix's pick on https://github.com/discopy/discopy/pull/817#discussion_r4375511919: A, colourless types as one-object categories, or B, no `None` for an unasked boundary anywhere
- [x] One helper for the generator round-trips, beside `assert_axioms`
- [x] `test_trace_repr` pins its literal
- [x] `decompositions` within three levels
- [x] `Spider.strategy` draws a phase
- [x] `Spider.__repr__` prints its phase as the argument `__init__` reads, where it printed `phase=` and did not read back: found by drawing phases
- [x] Constrained layers drawing free boxes only stays: proposal 2 of the description, answered on its thread
- [x] `assert_generators` draws single-shot with no example database, and checks each generator by its own `serialisation` and `transparency` laws (cubic's second pass)
