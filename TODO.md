# Property-based testing

## Current prompt

> don't push on the cmap branch, focus on proptesting. implement your changes not in fable bug investigation branch, moving bug reproductions from the test/fable/*.py files to test/test_properties.py, then once every relevant bug was classified and implemented as part of the general test suite, upstream the changes in the principal development branch for property testing

> feel free to implement new axioms in the missing categories

> the new property tests for this round should be test_properties.py, later we will investigate more complete test suite that does not only test categorical axioms but more general properties like roundtrips between different representations, string and pickle serialization, etc...

- [x] Fix the property suite: multi-box layers reached `Diagram.normal_form()` while
      `interchange()` only handles singleton-box layers, so `normal_form` now expands
      them into staircases first.
- [x] Integrate the categorical subset of [#606](https://github.com/discopy/discopy/issues/606)
      into `proptest/test_properties.py`. See **Classification** below for what landed,
      what became an axiom, and what does not fit the axiom matrix.

No verbatim prompt: this PR was opened 2026-06-22, before `RULES.md` rule 1 existed, and its
description is a generated summary rather than anything a human typed. It gets a `TODO.md` now
because USER ruled on [desire#76](https://github.com/toumix/desire/issues/76), verbatim:

> adoption adds a todo

That ruling is what unsticks this branch. `no-todo-on-main` marks a PR ready only when a push
*deletes* a `TODO.md`, so a branch that never had one can never leave draft, however green it is.
This file restores the normal gate: USER deletes it, the guard marks the PR ready.

## State

All three sign-off criteria otherwise hold, re-checked 2026-08-17: 0 behind `main` (`48eea53`),
CI green, no thread waiting on an agent. The Hypothesis matrix lives in `proptest/`, outside
pytest's `testpaths`, and runs on `main`, on manual dispatch, and on this PR via its `proptest`
label.

## Points

Cubic reviewed the branch on 2026-08-16 and raised ten findings. All are now settled below: eight
were fixed and two were false positives.

- [x] **The P1 is a false positive.** Cubic reads `Arguments.bifunctoriality` and friends as
      returning an unpacked N-tuple that `axiom(*arguments)` would splat onto a single structured
      parameter, raising `TypeError: too many positional arguments`. They do not: every one of them
      ends on a trailing comma, so they return a **1-tuple wrapping** the N-tuple and the splat
      passes exactly one argument.

      ```python
      >>> args = Arguments.bifunctoriality(monoidal.Diagram)
      >>> len(args), len(args[0])
      (1, 4)
      ```

      `test/abc.py` passes on this head. Had the claim been right the suite would have been red for
      eight weeks.
- [x] `all_axioms` binding each axiom twice was cosmetic and true; the generator now yields the
      already-bound axiom.
- [x] **The pivotal finding is a false positive.** `normal_form()` is called for its exception: it
      rejects planar diagrams that are not boundary-connected, while conversion intentionally
      preserves the original presentation. Two differently nested disconnected circles convert
      directly to the same hypergraph, and a regression test now documents that the public method
      rejects this unfaithful conversion.
- [x] `FeedbackCategory.feedback(self, dom, cod, mem)` declared all three arguments required even
      though both implementations and axioms infer them. The abstract signature now gives all three
      `None` defaults, matching `feedback.Diagram` and `stream.Stream`.
- [x] `Arrow.strategy` now applies exact boundaries to recursively generated paths, preserving
      `min_leaves` and `max_leaves`; a regression test finds a constrained composite.
- [x] `Layer.strategy` now filters candidate boxes through `exclude`, and `symmetric.Layer` forwards
      the set to its base distribution. Testing this also exposed and fixed its stale `Layer.cast`
      call, which broke unconstrained generation after `Layer.cast` was removed.
- [x] `CMap.strategy` now converts with `cls.from_diagram`, and a custom-subclass regression test
      checks the sampled value's exact type.
- [x] The global `"def strategy"` coverage exclusion is gone. Concrete strategy smoke tests keep the
      default suite at 98%; only the abstract `Strategy.strategy` declaration has a local pragma.
- [x] The tautological CMap axiom-name comparison now evaluates unitality and associativity on
      concrete maps; the property matrix continues to check every declared map axiom.
- [x] `test_extend_strategy` now finds both a free box and a twist from the extended distribution,
      while retaining its boundary-guard assertion.
- [x] Merged `main` (`5fa95f6e`), keeping both `CMap.from_generator` and the new
      `CMap.from_glued`, then took the simplifications the merged changes allow: the
      `axiom_status` entries that `Category.axiom_equality` already resolves through the MRO
      are gone, and `benchmark.generators.single_layer_tensor` builds its layer from boxes
      alone now that `monoidal.Layer` normalises its own plumbing.

## Round two: axioms carry their own verdict

> - `_typing` axioms are equating types, not diagrams, so they currently lift the types by constructing the identity. this is bad, axioms should work for any kind of returned equations.
> - i believe we should return to the original explicit call cls.equation_factory (or cls.ob.equation_factory, or ==), and remove the eq parameter.
> - remove `axiom_status`. instead, override the axiom and restate it to either return NotImplemented (wontfix), return AxiomError (bug), return an equation (over any carrier type, not necessarily Self, normal, setoid or strict)
> - using the above, integrate axioms into functors as well. axiom functors would test functoriality on arbitrary diagrams. equations would be not on Self, so the trailing eq parameter on axioms was not a good solution.

- [x] Bodies return their own verdict; `AxiomStatus`, `axiom_status`, `axiom_equality`,
      the `eq` parameter and the unused `strict` flag are gone.
- [x] The 79 status entries became overrides. `declared_axioms` is factored out so
      `CMap.axioms` honours what a map restates.
- [x] The `_typing` axioms and `self_dual` equate objects.
- [x] Functoriality axioms with an `Endofunctor` shape; `functor_spiders` catches #606's
      B16, the identity functor dropping a spider phase.

### Two things the design forced, worth a look

**A broken law is one with *some* counterexample, not every one.** `assert not
equation` per example was wrong: most draws satisfy a law that fails only for
particular inputs, e.g. `Matrix.copy` only for `x, n >= 2`. So `Axiom.broken` reads
the verdict off the body — `"AxiomError" in function.__code__.co_names` — before any
argument exists, and the matrix marks the test as an expected failure while the
search finds the counterexample. A fix shows up as XPASS rather than a hard failure;
making it loud means `strict=True`, at the cost of flaking whenever 25 draws miss
the counterexample.

**Restating an axiom duplicates its body**, which `STYLE.md` warns about, and it
leaves the `abc` statement dead where a class restates it. That is why the suite
dropped below the coverage gate until `test/abc.py` grew
`test_abstract_axioms_are_well_typed`. The 9 strict restatements are verbatim copies
of the abstract body; they only exist to undo a base class's `NotImplemented`, and a
way to say "the inherited statement, unchanged" would remove them.

### Deferred: functoriality on arbitrary diagrams

`Endofunctor` generates two free boxes, not arbitrary diagrams. Sending an arbitrary
diagram needs per-level knowledge the shape does not have: cups and caps need adjoint
types, feedback needs delayed ones, and the boundary-constrained image of a random
box is often unsatisfiable, which hung generation on symmetric and markov. Free boxes
exercise identity, composition and tensor at every level; structural preservation
(`F(swap) == swap`, `F(cup) == cup`) wants its own round.

## Classification of issue #606

The fable repros that state a categorical equation are now generated checks. The rest
either need a semantic carrier that does not exist yet, or are not equations at all.

| Finding | Where it landed |
|---|---|
| B1 `finset.Function.swap` inverse | `finset.Function` is a `SymmetricCategory` with a strategy; `hexagon_left`, `hexagon_right` and `braid_naturality` are `"bug"`. `Permutation` declares them `"strict"`, since its own composition convention is self-consistent — the bug is `Function.swap` reusing a permutation built for the opposite convention. |
| B5 `Matrix.copy` wrong for `x, n >= 2` | `Matrix` is a `MarkovCategory` with `ob = Natural` and a strategy; `Matrix[int]` joins the matrix and `copy_counitality`, `copy_cocommutativity` and `copy_monoidal_coherence` are `"bug"`. |
| B7 `Tensor.spiders(0, 0, Dim(n))` | New `HypergraphCategory.spider_fusion` axiom, `"setoid"` on `frobenius.Diagram`. The carrier that would go red is `Tensor`, see the blocked point below. |
| B25 heterogeneous-memory feedback | `FeedbackJoining` draws its two memory units independently and no longer pre-validates with the operation under test; `feedback_joining` is `"bug"` on `feedback.Diagram`. |
| B17 `ribbon.Braid.rotate` on daggers | **Not an equation.** `Diagram.rotate` builds with `_scan=False`, so a wrongly typed `Box.rotate` never reaches a boundary: both sides of any equation carry the same broken layers and compare equal. The new `rotate_contravariance` axiom holds regardless. B17 is a well-formedness property — `Diagram(d.inside, d.dom, d.cod)` raises — and belongs to the round that adds one. |
| B10, B11, B16, B18, B21, B29, B30 | Still need dagger, functor-law or operad abstractions that the matrix does not have. |

## Found on the way, not from #606

- `CMap.to_diagram` cannot convert back a map holding a traced box:
  `CMap.from_diagram(f.trace()).to_diagram()` raises `ValueError: 1 is not in list`
  for `f = symmetric.Box('f', a @ a, a @ a)`. `braid_naturality` is `"bug"` on
  `symmetric.CMap` and `closed.CMap`, whose equality goes through `to_diagram`.
  Worth its own issue.
- `ribbon.Twist(x).rotate()` has domain `x` where it should have `x.r`, the same
  shape of bug as B17 and reachable from the same fix.

## Blocked, worth its own pull request

`Tensor` is not declared as any abstract category, so B7 has no carrier that exhibits
it and B15's ZX evaluation has nowhere to land. Making it a `HypergraphCategory`
means a `Dim` strategy, a default `n` on `copy`, `ev`/`curry`/`rotate`, and integer
`swap` boundaries — 39 inherited axioms of which none pass today. That is a change to
`discopy.tensor`, not to the property matrix, and it should not ride on this branch.

## Answered question

> The `check_*` methods land in `discopy/abc.py`, so every category class in the library gains ~15
> test-only methods on its public surface. Defensible under `STYLE.md`'s "we expose the interface of
> every subprocedure as methods that can be tested and reused", and it is the one place where test
> scaffolding becomes API — raised on the PR 2026-07-25 and never answered. The alternative is
> Hypothesis strategies living in `proptest/` only. Worth a yes or no before this merges, since it is
> much cheaper to move now than after release.

USER answered **keep**, 2026-08-25.

The question was also stale as phrased: there are no `check_*` methods, `grep -rn 'def check_'
discopy/` finds only the unrelated `stream.py::check_later`. What the hierarchy actually carries is
the `Axiom` descriptors, a `strategy()` classmethod per class and the `axiom_status` dicts. Keeping
them is the right call for the same reason: the axioms are the specification of each category rather
than test scaffolding, and `axiom_status` is where the library records which laws each implementation
really satisfies. Moving them to `proptest/` would re-derive per-class dispatch outside the
hierarchy, which `STYLE.md` calls working at the wrong level of abstraction.
