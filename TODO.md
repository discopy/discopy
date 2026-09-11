# TODO

> remove @proptest/carriers.py and put everything into @proptest/test_axioms.py. don't filter based on `issubclass(Testable) and "strategy" in cls.__dict__`, instead you should manually override `def axioms` for every class that does not implement `Testable` yet, raising `NotImplementedError(f"No search strategy implemented for {cls.__name__}")`

- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 11:28 Declare the opt-out once in `discopy.axioms`, next to `Theory.axioms`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 11:28 Override `axioms` on every class that does not generate its own terms
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 11:28 Fold `proptest/carriers.py` into `proptest/test_axioms.py` and delete it
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 11:28 Update `CHANGELOG.md`, `CONTRIBUTING.md` and `AGENTS.md`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 11:28 `pflake8`, the test suite and the property matrix
