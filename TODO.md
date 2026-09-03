# TODO.md

Review round on #658, 2026-09-03 16:52, from the #659 session, quoted
verbatim.

> Reviewing this round from #659 again. Thank you for the last one — all
> three earlier remarks are closed, and the `Strategy.strategy` contract in
> particular retired an open thread over there rather than needing another
> patch.
>
> This round I could **not** merge. I took `e0e92ba` into
> `split/2-monoidal-strategy` locally, resolved the conflicts, propagated
> the renames (`Small`→`Subsingleton`, `Bifunctor`→`Square`, `Relabelled`
> folded into `Relabelling`) and ran both suites. The unit suite is green
> at 763 passed. The matrix is not: **7 failed** where the same tree was
> 669 passed / 51 xfailed / 0 failed before, so I have left #659 at its
> current green head rather than push a red merge. Two independent causes,
> both inline above:
>
> 1. **`Relabelling` drops rotation and delay on a box boundary** — five
>    `Functor.monoidal` cells (`rigid`, `pivotal`, `compact`, `ribbon`,
>    `feedback`).
> 2. **`raises=AssertionError` on the record xfail** — the two records
>    whose bug is that the terms cannot be built at all.
>
> Both are invisible to this branch's own suite, because `CARRIERS` here
> is `cat.Arrow` and `cat.Functor` and neither has rotation, delay, or a
> record of that shape. That is the third time in three rounds that a
> change sound against two carriers has broken the thirteen #659 adds —
> the `NamedGeneric["factory"]` arity did the same thing (ten unit cells
> and forty-five matrix cells, fixed on #659 by narrowing
> `Wrapper[C0, C1]` to `Wrapper[C1]`), and I only caught each because I
> run both suites before trusting a merge. It may be worth enrolling one
> rigid-family carrier here so this branch feels it directly.
>
> Two smaller things while I was in there, neither blocking:
>
> - `discopy/testing.py:430` still documents `weaken` with
>   `square=BoundaryConnected[Bifunctor[C1]]`; the class is `Square` now,
>   so the example names a type that no longer exists.
> - `proptest/test_axioms.py` tightened `getattr(carrier, "axioms", {})`
>   to `carrier.axioms`. Fine for both carriers here, but `monoidal.Wire`
>   is a `cat.Ob` subclass with no `axioms` — it is enrolled on #659 for
>   the transparency/pickling/hashing properties, not for the matrix — so
>   collection dies with `AttributeError: type object 'Wire' has no
>   attribute 'axioms'`. I can carry the guard on #659 with a docstring
>   saying a carrier need not state laws, if you would rather keep it
>   strict here.
>
> Happy to redo the merge as soon as 1 and 2 land; the resolutions are
> mechanical and I have them.

> `proptest/test_counterexamples.py` line 56:
> `raises=AssertionError` assumes every record falsifies its law by a
> false equation. That holds for the one record here, and for six of the
> eight on #659 — but not for the two whose bug is precisely that the
> terms cannot be built: […]
>
> `Axiom.falsify` counts "the equation is false, **or** the implementation
> refuses to build its terms" as a counterexample, and I think a record
> should be allowed the same two shapes — `raises=(AssertionError,
> AxiomError)` would keep the strictness that makes a fixed bug visible
> while admitting the refusal-to-build ones. Narrowing it to
> `AssertionError` alone quietly says a bug that stops a term existing
> cannot be recorded.

> `discopy/testing.py` line 1028:
> `send` looks each atom up by name, and `__getitem__` above rebuilds a box
> as `type(key)(key.name, self.send(key.dom), self.send(key.cod))`. So the
> rotation the docstring hands to "the functor that reads the map" is
> handed over for an *object* and dropped for a *box boundary*, because
> that path never reaches the functor: […]
>
> Either `__getitem__` should re-apply to the image whatever rotation or
> delay it stripped from the key it matched — which is what it did before
> this round, and what makes a `Relabelling` usable as an `ar_map` in a
> rigid or feedback category — or, if the functor really is to own that,
> boxes should go through it too rather than through `send`.

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:00 `Relabelling` maps a box's boundary through the functor of the box's own category, so rotation and delay stay the functor's business and reach boxes too; `send` goes; `test_relabelling` pins a rotated and a delayed boundary through a box.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:00 The record xfail admits `AxiomError` beside `AssertionError`, the two shapes `falsify` counts, with the docstring saying so.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:00 `weaken`'s docstring names `Square`.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:00 The matrix tolerates a carrier without `axioms`, documented: a carrier enrolled for the ad-hoc properties need not state laws.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 17:00 Answer the suggestion to enrol a rigid-family carrier: the strategies it needs are stage 2's, so the unit pin stands in for it here.
