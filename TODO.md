# TODO

> the property testing infrastructure PR has been merged. extract the serialisation methods from `Testable` and instead put it in `Serialisable`, which should be moved to `abc`. define the relevant serialisation roundtrip axioms and test them in the new property testing suite.

- [ ] merge `main` into the branch: `cat.Ob`/`Arrow` take both `strategy` and `serialised_attrs`, and the `NamedGeneric` pickle fix follows `NamedGeneric` down to `utils`
- [ ] move `Serialisable` from `utils` to `abc`, `BinaryBoxConstructor` declaring its attributes without the base
- [ ] extract `environment`, `transparency`, `pickling` and `serialisation` from `axioms.Testable` into `Serialisable`, leaving `Testable` the generation contract alone
- [ ] state the roundtrip laws as axioms of `Serialisable`, one per mechanism
- [ ] enrol the serialisable carriers in `proptest/` and check the laws run there
- [ ] `CHANGELOG.md` entry, `pflake8`, `pytest` and the `proptest` suite green
