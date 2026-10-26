> toumix, 2026-10-26, on `monoidal.is_monochrome`: "Code smell: this None shouldn't be here, this whole method shouldn't exist" and "That's not even what monochrome means ffs"
>
> cubic, 2026-10-26, requested by toumix: the generator round-trip loop is copied into 14 test modules; `test/traced.py` derives its expected repr from `repr(f)`; `decompositions` nests four loops deep; `Spider.strategy` never draws a phase; constrained layers only draw free boxes.

- [ ] `is_monochrome` goes, waiting on toumix's pick on https://github.com/discopy/discopy/pull/817#discussion_r4375511919: A, colourless types as one-object categories, or B, no `None` for an unasked boundary anywhere
- [WIP] @session_01F6hn2cPwGcF56EfWivjiDo-2026-10-26 09:10 One helper for the generator round-trips, beside `assert_axioms`
- [WIP] @session_01F6hn2cPwGcF56EfWivjiDo-2026-10-26 09:10 `test_trace_repr` pins its literal
- [WIP] @session_01F6hn2cPwGcF56EfWivjiDo-2026-10-26 09:10 `decompositions` within three levels
- [WIP] @session_01F6hn2cPwGcF56EfWivjiDo-2026-10-26 09:10 `Spider.strategy` draws a phase
- [x] Constrained layers drawing free boxes only stays: proposal 2 of the description, answered on its thread
