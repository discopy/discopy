# test/fable — the red suite of the second read

One file per finding of [#606's second read](https://github.com/discopy/discopy/issues/606#issuecomment-5371739173):
`B<n>.py` reproduces bullet B*n*, `P<n>.py` is a deterministic miniature of property P*n* over
curated examples. Every test asserts the **correct** behaviour, so it **fails while its bug is
live** and passes once fixed — the directory is an acceptance harness, not part of the green suite.
Two deliberate passing controls: B6's `left=False` trace and B13's `CRz` translation.
Some tests carry a passing sub-assert as its own control (e.g. B30's graft composite building).

Run one file or all of them (note the glob — a bare directory argument collects nothing, since
these filenames don't match pytest's default `python_files`):

    uv run pytest test/fable/B5.py -q
    uv run pytest test/fable/*.py -q

Verified on `main` at `107c846`: 104 tests, 102 failed (the bugs), 2 passed (the controls), in
under four seconds. `B24`, `B40`–`B44` and parts of `B45` are static checks on sources and config,
marked as such in their docstrings; `B24` is static because torch is not installable here, `B30`'s
dependency check uses a stub token so spaCy is not needed.

| ID | Claim |
|---|---|
| B1 | `finset.Function.swap` returns the inverse permutation |
| B2 | `python.additive` trace feeds back on the wrong wire when dom'/cod' lengths differ |
| B3 | `python.multiplicative` trace raises under default type checking; `trace(n>1)` unconstructible |
| B4 | `finset.Function.permutation` ignores block structure |
| B5 | `Matrix.copy(x, n)` fails counit and cocommutativity for x, n ≥ 2 |
| B6 | `Matrix.trace` ignores `left` |
| B7 | `Tensor.spiders(0, 0, Dim(n))` returns 1, breaking spider fusion |
| B8 | `Tensor.cup_factory` wrong on non-atomic dims |
| B9 | `Matrix.map` casts to the old dtype; composition drops the dtype |
| B10 | `Controlled.array` ignores dagger/conjugate — `Controlled(S).dagger()` is a numeric no-op |
| B11 | `subs`/`lambdify` drop `distance` and `is_mixed`; `Sqrt.dagger` unconjugated |
| B12 | `U1.grad(mixed=False)` disagrees with the finite difference |
| B13 | `gate2zx` mistranslates CRx and CU1 |
| B14 | `to_pyzx` silently turns Y spiders into X spiders |
| B15 | Inherited `.eval()` broken for every zx diagram |
| B16 | `frobenius.Functor` drops spider phases; phased `Spider` repr not evaluable |
| B17 | `ribbon.Braid.rotate` wrong for dagger braids |
| B18 | rigid `Box.dagger()` silently resets `z` |
| B19 | Oversized trace silently succeeds; negative trace recurses |
| B20 | `Channel[float].discard`/`cups` lose the dtype |
| B21 | `Bubble.dagger()` crashes; `Trace`/`Curry`/`Twist` fail `loads(dumps(...))` |
| B22 | Eval crashes on `Measure(override_bits=True)` |
| B23 | `to_tk` dies on CCX/CY/CRy/CU1; `from_tk` crashes on U1/CU1 |
| B24 | pennylane uses removed `Operation.inv()`; scales by `_scale ** 2` (static) |
| B25 | Heterogeneous-memory `feedback` broken; `feedback.Diagram.discard` uninitialised |
| B26 | `closed.Substitution` returns None on constants, recurses on abstractions |
| B27 | `Id(x).width` raises on empty `max()` |
| B28 | `Stream.permutation` broken for non-identity inputs |
| B29 | `Drawing.validate_attributes` is a TypeError; `Drawing.dagger()` asserts on ≥2 boxes |
| B30 | `grammar.cfg` unhashable/uncomparable; dependency leaves never become `Word`s |
| B31 | `pregroup.normal_form` crashes on foliated and left-whiskered diagrams |
| B32 | `tensor.Box` equality/hash raise on array data |
| B33 | `to_tn` crashes on bit discards and fancy measures; classical `measure()` crashes |
| B34 | `rigid.nesting` never checks the two types have equal length |
| B35 | Failing `from_callable` leaks a monkey-patched `__call__` |
| B36 | `Matrix.__repr__` mutates numpy's global printoptions and elides > 16 entries |
| B37 | `draw()` mutates the drawing it draws |
| B38 | zx `H`'s instance-lambda dagger makes diagrams unpicklable |
| B39 | `eval(repr(x))` fails across `traced`/`cat`/`interaction`/`channel`/`quantum` |
| B40 | `--skip-extra` contract broken; pyzx guards ignore the version pin (static) |
| B41 | Coverage gate counts the test files themselves (static) |
| B42 | Tests that cannot fail in `test/cat.py`, `test/markov.py`, `test/readme.py` (static) |
| B43 | Nine modules have no test file; `test/test_para.py` naming drift (static) |
| B44 | Quantum interop tests are phase-0 and structural-only (static) |
| B45 | Runnable doc/message nits: self-contradictory errors, stale pointers, missing exports |
| P1 | Transparency round-trip: `eval(repr(x)) == x` and `loads(dumps(x)) == x` per box class |
| P2 | Dagger and rotation laws |
| P3 | Concrete-category laws: comonoid, swap naturality, trace, spider fusion |
| P4 | Numerical unitarity and gradients vs finite differences |
| P5 | `subs`/`lambdify` preserve every non-data attribute |
| P6 | The identity functor is the identity on every box |
| P7 | Representation round-trips evaluated numerically at non-zero phases |
| P8 | Purity: no observable global state from repr/draw/eval/`from_callable` |
| P9 | Every public op returns a well-typed result or raises `AxiomError` |
