# TODO

> Check out the hopf module in DisCoPy and the pivotal PR
> https://github.com/discopy/discopy/pull/484. The tests of the hopf module
> are currently very expensive in CI as there are big tensor contractions.
> Open a new PR proposing which tests can be removed or simplified, keeping
> the coverage while making the tests lighter. Write only the TODO.md and PR
> description so I can review before we make changes.

Measured baseline: `uv run pytest test/hopf.py discopy/hopf.py --durations=0`
gives 43 items (31 tests + 12 doctests) in 32.2s on main, times three Python
versions under coverage in CI. The 31 test functions alone give 100% statement
coverage of `discopy/hopf.py`; nine tests each own at least one line (error
paths, `qdim`, `__hash__`, `is_commutative`/`is_cocommutative`, the raw-array
`Intertwiner` branch) and are untouched below.

## A. `test/hopf.py` and doctests on `main` (32.2s, target ~5s)

- [x] `test_double_of_sweedler` (15.7s, half the suite): replace
  `Double(sweedler()).is_valid()` — whose `has_antipode` contraction hits an
  `n^14` intermediate under numpy's greedy einsum path — by the same axiom
  equations checked on the materialised `16^3` structure constants with
  `np.einsum`, as `test_double_antipode_inverse` already does. Diagram-level
  `is_valid()` of a double stays covered by `Double(cyclic(2))`.
- [x] Drop the `is_quasitriangular()` calls made right after `is_valid()` in
  `test_double_is_quasitriangular_hopf_algebra`, `test_double_of_sweedler`
  and the `Double` doctest: `is_valid()` already runs it, each call is a
  duplicate ~0.3s contraction.
- [x] `test_double_is_quasitriangular_hopf_algebra` (3.0s): drop `n = 3` —
  `Double(cyclic(3))` runs the same code paths as `Double(cyclic(2))` at
  several times the contraction cost.
- [x] Merge `test_functor_returns_a_tensor_network`,
  `test_nontrivial_link_invariant` and
  `test_crossing_number_distinguishes_closures` (mutual coverage subsets)
  into one test contracting each network once: circle 2, unlink 4, Hopf
  link 0, plus the unknot 2 as a single braid closure. Today the Hopf-link
  value is contracted four times across tests and the module doctest.
- [x] Remove `test_reidemeister_moves`: R2 is braid invertibility and R3 is
  Yang–Baxter, both already asserted on the same matrices by
  `test_braiding_yang_baxter_and_inverse`; its coverage is a strict subset
  of the remaining functor tests.
- [x] Trim duplicate `is_module()` contractions: the e⊕m module axiom is
  checked four times at identical dimensions across
  `test_representation_is_module`, `test_tensor_of_representations` and the
  `Representation`/`Representation.direct_sum` doctests — keep one per
  distinct code path (product action, trivial factor, direct sum, anyons).

## B. New tests of PR #484 (to apply on `pivotal-structure`)

Measured on the branch: `test/hopf.py` runs 37 tests in 47.2s; the six new
tests add ~15s of which 13.7s is `test_ribbon_element_criterion` alone, the
other five are each under 0.6s.

- [x] `test_ribbon_element_criterion` (13.7s): the cost is not the test but
  the elements' implementation — `drinfeld_element`, `pivotal_element` and
  `is_ribbon` each call `self.mult.eval` / `self.antipode.eval` again, so
  the same composite structure maps of `Double(taft(3))` are contracted
  several times at dimension 81 within one property access. Evaluate each
  structure map once and reuse the arrays in `hopf.py` (a #484 code tweak);
  keep the test as is. Done as the cached `Algebra.arrays`, though profiling
  showed the dominant cost (22s of 24s) was elsewhere: the pivotal-element
  SVD materialised the discarded `6561^2` factor `U`, fixed by
  `full_matrices=False`. 19.9s down to 0.8s.
- [x] Share one module-level `Double(sweedler())` between
  `test_pivotal_element` and `test_ribbon_element_criterion` and the
  `taft(3)` / regular representation / `Functor` setup between
  `test_pivotal_pairings_are_intertwiners` and
  `test_snake_equations_with_pivot`, so `cached_property` reuse kicks in
  (~1s combined).
- [x] Keep the Kauffman–Radford pair itself unchanged: `Double(taft(3))`
  ribbon and `Double(sweedler())` not ribbon is the smallest instance of the
  odd/even criterion, it cannot be exercised below dimensions 81/16.

## C. Observations to file as issues (not test changes)

- [x] The `has_antipode` blow-up is a contraction-path pathology, not an
  inherent cost: `np.einsum(..., optimize="greedy")` in
  `tensor.Diagram.eval` picks an `n^14`-scaling path on the 53-box antipode
  network (`optimize="optimal"` or `opt_einsum` finds a cheap one). Fixing
  the path selection would make `Double(sweedler()).is_valid()` fast and
  section A's first point unnecessary — decide which of the two to do.
  Filed as [#595](https://github.com/discopy/discopy/issues/595).
- [x] Latent failure path: networks beyond `config.MAX_EINSUM_INDICES = 52`
  fall back to `opt_einsum`, which is not a required dependency.
  Filed as [#596](https://github.com/discopy/discopy/issues/596).
