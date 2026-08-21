# TODO

USER review on #594, 2026-08-20, verbatim.

On `discopy/quantum/circuit.py`:

> Actually this is a good point: SWAP is a real quantum gate which should count
> as a box, not plumbing. Let's make it *not* a subclass of Swap (it can still
> have draw_as_swap=True) so that we can distinguish logical vs physical swaps

On `test/cmap.py`:

> weird? the previous version looked weaker not stronger: how come to_map
> generates two swaps that cancel each other?

> waiiiit a minute: is this test confusing imperative vs functional? .swap is a
> class method that returns a swap so stacking two like this makes no sense
> whatsoever, .permute is what the tests should be doing
>
> let's scan through this test file and make sure we fix it

- [x] Fix `test_eliminate_swaps` and scan
  `test/cmap.py` for classmethods called on an instance: `Id(x @ y).swap(x, y)`
  discards its receiver, so the test composed no swaps at all and eliminating
  them was vacuous. Use `permute` and assert the swaps really are eliminated.
- [x] Make the physical `SWAP` gate a box rather than the categorical swap, so
  that logical and physical swaps are distinguishable.

USER 🚀 on daydream6728's three review comments, reacted 2026-08-19 onwards:

- [WIP] @session_01CxwLYQPPYJ4pgsJ77UEVWG-2026-08-21 13:25 Annotate
  `python/additive.py`'s `permutation` with `-> Self`.
- [WIP] @session_01CxwLYQPPYJ4pgsJ77UEVWG-2026-08-21 13:25 Define `permutation`
  once on `finset.Function` so `python.additive` inherits it.
- [WIP] @session_01CxwLYQPPYJ4pgsJ77UEVWG-2026-08-21 13:25 Show the foliation of
  either side in the middle of the yang-baxter equation drawing: a single
  `Permutation` box `[2, 1, 0]`.
