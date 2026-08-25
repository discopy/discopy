Thanks Sonnet but I don't think you answered the question and I don't
think your recommendation makes sense, let's merge this and we add the
reverse rule refactor in a second step;

USER's instruction on #575, after #575 merged into `main` with `copar` kept.
"This" = #575 (merged as 1a90774), "the reverse rule refactor" = the
follow-up noted on this branch since #571 opened: refactor
`neural.rdiff.ReverseRule.then`/`tensor` onto `para.Symmetric`'s now-general
composition, which #559/#572/#575 built for exactly this purpose.

Mathematical design, checked before writing code: `ReverseRule` pairs a
forward leg `A -> B @ M` with a reverse leg `M @ B -> A`. The forward leg is
literally a `Copara` (`para.Symmetric` with empty `param`, `copar = M`):
its hand-rolled `then`/`tensor` swap arithmetic is byte-for-byte the
empty-`param` case of `Symmetric.then`/`tensor`, checked by hand against the
code in `para.py`. The reverse leg does **not** reduce the same way: its
domain is `residual @ cod` — residual on the *left* — the mirror of
`Symmetric`'s `dom @ param` convention. Composing it as a plain `Para` would
need `cod @ residual` instead, and that flip is not a wash: the *current*
`residual @ cod` order is what lets `ReverseRule.__init__` infer `cod` from
`forward`/`reverse` alone (the `candidates` search only disambiguates
because the swapped form differs from `forward.cod`; the unswapped
`cod @ residual` form always trivially equals `forward.cod` for any split,
so the inference would degenerate and every caller would need to pass `cod`
explicitly). So the reverse leg's convention is load-bearing, not an
oversight — matching it to `Symmetric` would either reinsert the same swap
arithmetic we're trying to remove, or cost real functionality. This
confirms, precisely, what #571's original "leave `rdiff` alone" note got
right and what it got wrong: right that a naive full refactor is a bad
trade, wrong that *no* part of it is worth doing — the forward leg alone is
a clean, free win.

- [ ] Refactor `ReverseRule.then`/`tensor`'s forward-leg computation to
      delegate to `para.Symmetric` (empty `param`, `copar` = the residual)
      instead of hand-rolled swap arithmetic; leave the reverse leg exactly
      as written, with an explanation of why in the module docstring near
      the residual-order note.
- [ ] Keep `ReverseRule`'s public constructor, fields and stored diagrams
      unchanged — this is an internal implementation change, not an API
      change, so `differentiate`/`rdiff`/`benchmark/catgpt.py` need no edits.
- [ ] Test that composed forward legs match `Symmetric`'s `then`/`tensor`
      structurally, run the full CatGPT conformance suite (forward,
      structural VJP against autograd, one SGD step) to confirm the
      refactor is behaviour-preserving, not just type-preserving.
- [ ] `pflake8 discopy`, full suite, `CHANGELOG.md`.
