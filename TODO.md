# TODO

Review round, cubic 2026-08-26: [8 open threads on #347](https://github.com/discopy/discopy/pull/347)
and the branch is in conflict with `main` (`mergeable_state: dirty`). Feedback quoted per box,
verbatim in each linked thread. Fix or refute with evidence, resolve the thread either way;
delete this file when the round is done.

- [x] merge `main` into the branch (append-only, never rebase), rerun `pflake8` and `pytest`
- [WIP] @evening-2026-08-27T20:05Z [testing.py:505](https://github.com/discopy/discopy/pull/347#discussion_r3863535157) P2 — "`Axiom.broken` is computed by scanning `function.__code__.co_names` for the literal string `\"AxiomError\"` … a fragile heuristic": set an explicit flag in `.failing()` instead
- [WIP] @evening-2026-08-27T20:05Z [cat.py:1016](https://github.com/discopy/discopy/pull/347#discussion_r3863535169) P2 — "`Functor.strategy` declares `dom` and `cod` keyword parameters but never uses them": honour them or drop them
- [WIP] @evening-2026-08-27T20:05Z [proptest/test_axioms.py:65](https://github.com/discopy/discopy/pull/347#discussion_r3863535177) P2 — "A broken axiom (xfail) is only exercised if one of the 25 random samples happens to be its counterexample": search with `hypothesis.find` or make the xfail strict
- [WIP] @evening-2026-08-27T20:05Z [abc.py:101](https://github.com/discopy/discopy/pull/347#discussion_r3863535188) P2 — "docstring … names both as `cls.equation_factory`": name the quotient factory it means
- [WIP] @evening-2026-08-27T20:05Z [CHANGELOG.md:59](https://github.com/discopy/discopy/pull/347#discussion_r3863535201) P2 — "'Added' entry describes a strict/setoid axiom-status system as if it shipped" while 'Changed' removes it: fold into 'Changed'
- [WIP] @evening-2026-08-27T20:05Z [monoidal.py:840](https://github.com/discopy/discopy/pull/347#discussion_r3863535212) P3 — "`Layer.strategy` accepts `boundary_connected` but never uses it": wire it in or drop it from the signature
- [WIP] @evening-2026-08-27T20:05Z [matrix.py:142](https://github.com/discopy/discopy/pull/347#discussion_r3863535227) P3 — "`Matrix.strategy()` always draws entries as non-negative integers … ignoring `cls.dtype`": generate entries per dtype
