# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
session, per the discopy#513/#514 precedent (see also optyx#34, discopy#537).

Fixes #534: `discopy/quantum/circuit.py` imports the array-backend context
manager from `discopy.matrix` under the name `backend`, the same name
`Circuit.eval` and `Circuit.get_counts` use for their own pytket `backend=`
parameter. Inside those two methods' bodies the import is shadowed and
unreachable; today neither method's own body happens to call it, so there
is no live crash, but any future code added inside `eval`/`get_counts` that
needs to switch array backends would silently hit the wrong object.

- [x] Rename the import to `array_backend` and update its three call sites
      (`Circuit.measure`, `Circuit.to_tn`, `Box.array`) — a pure rename,
      the public documented `eval(backend=...)`/`get_counts(backend=...)`
      pytket API is untouched.
- [x] Add a regression test (`test_array_backend_not_shadowed_by_eval_backend_param`
      in `test/quantum/circuit.py`) exercising `array_backend` alongside an
      `eval(backend=...)` call.
- [x] `CHANGELOG.md` entry added.
- [x] `uv run pflake8 discopy`: clean.
- [x] Manually verified the exact regression-test logic
      (`with array_backend('numpy'): (X >> X).eval(backend=None).array`)
      via `uv run python -c ...`: returns the expected `(2, 2)` array.
      Could not run `test/quantum/circuit.py` itself or
      `uv run coverage run -m pytest` in this sandbox — `uv sync --group
      quantum` fails fetching `torch` (`download-r2.pytorch.org`
      unreachable here, the same sandbox limitation noted in discopy#533)
      and the whole file `importorskip`s on `torch`/`sympy`/`tensornetwork`
      at module level. Needs a run on a networked machine or CI before
      sign-off.
