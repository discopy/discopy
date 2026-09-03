# TODO.md

> go ahead and implement the Equation-valued properties proposal

The proposal, as put on #658: state transparency, pickling and
serialisation as `Equation`-valued laws of every carrier — axioms on
`Strategy`, the type that generates its own instances — so the matrix
owns them and `proptest/test_repr.py`, `test_pickle.py` and
`test_serialisation.py` go; the environment a representation evaluates in
becomes a `Strategy` classmethod the carrier owns; the per-carrier
expected failures those files carried become `.failing` declarations on
the carriers, which live on #659.

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:20 `Strategy.axioms` collects the laws a type states or inherits, defined once and shared with `discopy.abc.Category`, so a carrier that is not a category has cells too.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:20 `Strategy.transparency`, `pickling` and `serialisation` as laws of an element, with `Strategy.environment` for the namespace a representation reads back in; `Axiom.strategy` resolves a law of an element on a carrier that is no category.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:20 The carriers of this branch: `cat.Functor` declares `serialisation` inapplicable, `testing.Natural` too, `Relabelling` reprs qualified by module so a generated functor reads back from the package namespace alone.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:20 `proptest/test_repr.py`, `test_pickle.py` and `test_serialisation.py` deleted; the module docstring, `CHANGELOG.md` and the pull request describe the laws; `test/testing.py` pins the three laws on a toy.
