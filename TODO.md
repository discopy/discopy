# cubic-dev-ai review round

cubic-dev-ai posted 7 unresolved findings on this PR. Verify each against the
actual code, fix if real, decline with a reason otherwise.

- P2 `discopy/neural/backend.py:78` — Concurrent `backend()` calls share
  `_stack`, so one thread/task can change another call's implicit backend
  selection. Store stack state in a `ContextVar` instead of a mutable
  default argument.
- P2 `benchmark/catgpt.py:589` — `random_batch` never samples the last valid
  shifted window because `torch.randint` excludes its upper bound. Increase
  the upper bound by one valid index.
- P2 `discopy/neural/network.py:683` — A negative `n_rounds` silently runs
  zero rounds and returns the initialized messages, hiding caller bugs.
  Reject negative values before entering the execution loop.
- P3 `benchmark/test_catgpt.py:133` — The "non-wrapping" verification is
  vacuous: the synthetic stream is exactly periodic with period 7 == vocab,
  so the assertion passes for every window regardless of wraparound. Use a
  stream that cannot make this collapse.
- P3 `benchmark/test_catgpt.py:36` — Exact `==` against a hard-coded float
  literal is dtype-fragile (depends on the default dtype at scalar
  promotion). Compare with `pytest.approx`.
- P3 `discopy/neural/rdiff.py:154` — Catching `(KeyError, TypeError)` around
  a rule lookup rewrites a caller's real `TypeError` into a misleading
  "Missing reverse rule" `ValueError`. Catch only `KeyError`.
- P3 `benchmark/catgpt.py:583` — When `len(stream) == context + 1` there is
  one valid shifted window, but the length guard rejects it. Allow the
  minimum valid length.

Checkboxes:

- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 backend.py:78 thread-unsafe `_stack`
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 catgpt.py:589 randint off-by-one
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 network.py:683 negative n_rounds
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 test_catgpt.py:133 vacuous wraparound test
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 test_catgpt.py:36 float `==` comparison
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 rdiff.py:154 overly-broad except
- [WIP] @agent-a4c984f94804460c5-2026-08-25 00:00 catgpt.py:583 off-by-one length guard
