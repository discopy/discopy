# test/fable — the red suite of the full reads

One file per finding of [#606's second read](https://github.com/discopy/discopy/issues/606#issuecomment-5371739173)
(B1–B45, P1–P9) and of [#699, the third read](https://github.com/discopy/discopy/issues/699)
(B46–B88, P10–P13): `B<n>.py` reproduces bullet B*n*, `P<n>.py` is a deterministic miniature of
property P*n* over curated examples. Every test asserts the **correct** behaviour, so it **fails while
its bug is live** and passes once fixed — the directory is an acceptance harness, not part of the green
suite. Deliberate passing controls carry `control` in their name or docstring.

Run one file or all of them (note the glob — a bare directory argument collects nothing, since
these filenames don't match pytest's default `python_files`):

    uv run pytest test/fable/B5.py -q
    uv run pytest test/fable/*.py -q

Verified on `main` at `ce044a0` (2026-09-02): the first read's 104 tests are 99 failed / 5 passed
(B2, B28's permutation and B43's naming went green since `107c846`); the third read's 238 tests are
220 failed / 18 passed controls, in ten seconds. `B24`, `B40`–`B44`, `B87`, parts of `B45` and `B88`
are static checks on sources and config, marked as such in their docstrings; torch is not installable
here, so the files needing pytket, pyzx, tensornetwork, jax or nltk `importorskip` them.

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
| B46 | `Tensor`/`Matrix` composition casts to the left operand's dtype: `id >> v` is zero |
| B47 | A cup on a non-atomic image differs alone and in a diagram: the tensor functor is not a functor |
| B48 | `Hypergraph.rotate` keeps dom and cod unswapped; `f.r.to_hypergraph()` raises, frobenius `simplify` transposes |
| B49 | `foliation()` runs snake removal through `Layer.merge`, and crashes on the README snake |
| B50 | Hypergraph equality and hash ignore scalar spider types |
| B51 | `balanced.Functor` and `DualRail` drop the dagger of a `Twist` |
| B52 | `hopf.Functor` sends the twist to the inverse of the braid's trace |
| B53 | `Representation.l` pairs on the wrong side of the cup |
| B54 | `Controlled.l`/`.r` are the conjugate, not the transpose; `CRz.rotate()` crashes |
| B55 | `circuit.Box(is_mixed=False).dagger()`/`rotate()` come back mixed |
| B56 | `to_tk` drops a permutation over mixed bit/qubit types |
| B57 | `to_tk` drops `Discard(bit)` |
| B58 | `gate2zx` checks gate names before `distance` |
| B59 | `Scalar.grad` drops `is_mixed` |
| B60 | The identity functor is not the identity on `Permutation` |
| B61 | `foliation`/`to_staircases`/`Functor.id` drop bubble names and drawing flags |
| B62 | `NamedGeneric[...]` loses its parameter through pickle |
| B63 | `rmap` on ndarray data, string data recursion, the `subs` list form |
| B64 | `to_tn` reads a 0-input spider's dimension from the wrong wire |
| B65 | `CMap.trace` never type-checks the traced wires |
| B66 | `Hypergraph.depth` undercounts paths starting at a state |
| B67 | Grammar `Word`/`Rule` daggers, `fc`/`bc` on non-atomic types, `from_nltk` truncation |
| B68 | `Channel.then` checks the flattened `Dim` only; `CQ.__str__`; `Ket(True)` |
| B69 | Daggered braids, delayed boxes and head/tail types do not survive `loads(dumps())` |
| B70 | `closed.Eval` defaults right-handed; oversized `Curry`; contradictory `Eval(left=)` |
| B71 | `CMap.to_diagram` places every state at the right; `make_causal` over-cuts beside a loop |
| B72 | Whiskered swaps crash `Matrix` functors, `to_braided` and PRO-widening functors (#594) |
| B73 | `to_staircases` uses `monoidal.Functor.id`: `foliation`/`depth` crash on traces |
| B74 | `Hypergraph.caps` has the cup convention: every rigid cap is rejected, pregroup foliation crashes |
| B75 | `make_causal` types the cut wire with the unwound spider type |
| B76 | No `bubble_factory` on eleven levels; `frobenius.Bubble` has no `z` |
| B77 | `interaction`: `caps` summands reversed, `trace`/`curry` stubs, `Int(Matrix[bool]).braid` |
| B78 | `braided.simplify` on a foliated diagram; balanced `Trace.to_braided` |
| B79 | `Bubble.subs`, dict `colour_map` on empty types, `substitute`/`grad` on foliated input |
| B80 | `Matrix` under jax, tuple-typed wires, `Functor(dtype=None).__repr__`, `Channel.conjugate` |
| B81 | Categorial `FC`/`BC`/`FX`/`BX` are not boxes; nested `cfg.Tree` call; `to_compact` into a dagger-less functor |
| B82 | `(f + g).to_map()`, `QuantumGate` data, string-phase `Spider.dagger`, `to_rigid` drops data |
| B83 | Non-named colours crash spiders, ribbons and twists; TikZ `->looseness` and `scale=` |
| B84 | `post.py` dying before `record()` hands over as clean; `benchmark_comment` on a deleted fork |
| B85 | `frame_dual_rail` is not idempotent; `Drawing.then`/`tensor` alias; figure leak |
| B86 | Transparency: `cat.Bubble`, `Eval`/`Coeval`, `pregroup.Word`, `cfg`, `FollowedBy`, `CMap`, unhashable dataclasses |
| B87 | Suite: circuit tests skipped whole without torch, global `H` mutated, tests that cannot fail or assert the bug (static) |
| B88 | Docs, messages and config nits, and the `abc`/`para`/`Layer` signature drift |
| P10 | Foliation is an equality and `depth()` never raises |
| P11 | Encodings agree with a numeric oracle |
| P12 | Converters agree: `eval`, `to_tk`, `to_pyzx`, `to_tn` |
| P13 | Levels conform to `abc` and every level's bubble composes |
