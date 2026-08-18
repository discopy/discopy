# TODO

Prompt ([#444](https://github.com/discopy/discopy/issues/444), verbatim):

> `Swap` should be a subclass of `Permutation`, with a `__new__` on
> `Permutation` that catches the transposition `(1, 0)` on two wires and
> returns a `Swap`. This removes the special-casing between the two and opens
> the way to storing every even slot of a `symmetric.Layer` uniformly as a
> `Permutation` ("everything is a permutation").

USER (interactive session, 2026-08-18, verbatim):

> there shouldn’t be any regenerated svgs please fix that
> add the TODO: the PR was created in a synchronous session with ChatGPT and
> I told it not to add it because i thought it was gonna be published as my PR
> but actually it’s better as an agent PR

- [x] Make `Swap` a subclass of `Permutation`, with a `__new__` on
  `Permutation` catching the two-wire transposition (opening commit).
- [x] No regenerated drawing baselines: a `Swap` stays a generator rather
  than plumbing — no coalescing into wider permutations on whiskering,
  `from_box` keeps the swap drawing — and every `docs/_static` and
  `test/drawing/tikz` baseline reverts to `main`.
- [x] cubic P2 on `feedback.py`:
  `Permutation` inherits `Box.delay` which crashes with `TypeError`;
  override `delay` on `feedback.Permutation` and drop the special case in
  `feedback.Layer.delay`. `reset` crashes the same way but predates this PR
  (`feedback.Swap.reset()` fails on `main`) and is out of scope here —
  USER, 2026-08-18.
- [x] cubic P2 on `quantum/tk.py`:
  `current.index(source, i)` makes `to_tk` quadratic in the width of a
  permutation box; keep an inverse position table so each lookup is
  constant-time.
- [x] cubic P3 on `quantum/circuit.py`:
  the two swap fast paths in `to_tn` are unreachable behind the new
  `isinstance(box, Permutation)` branch; reorder so the equality checks
  come first.
- [x] USER on `drawing/drawing.py:547`:
  remove `Drawing.swap` unless something needs it, or explain why on the
  thread.
- [WIP] @claude-r6h92m-2026-08-18 15:10 USER review on `symmetric.py:568`
  (2026-08-18): a `Swap` should count as plumbing, breaking a bunch of
  stuff; the swap-box representation, if wanted, is a downcast to a
  monoidal diagram. 🚀 with "remove unnecessary labels": swaps coalesce
  like any permutation, `from_box` stops aliasing a permutation's cod with
  its dom so labels stay put, a wire at a fixed point of a band is not
  re-labelled, and the baselines are re-committed pixel-identical.
