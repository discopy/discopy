# TODO.md

> > this is not acceptable, a simple linter can avoid that mistake, why can't a frontier model do it? how do we prevent this from happening again? we've stopped taking pylint seriously but maybe we should? @daydream6728 any opinion on this?
>
> just had a look at the current linter output, its not pretty. we'd have hundreds of diagnostics to triage, either by disabling checks globally or disable on individual pieces of code. If we ever setup a CI job for that (which is ambitious in the current state), I'd also advise to not set a threshold number and instead force diagnostics to be explicitly disabled, i find it better for reviews.
>
> in any case i am curious to understand how come the property testing didn't violate any of the axioms you've defined. did it cheat and implement an incomplete search strategy?

— daydream6728 on [discopy#767](https://github.com/discopy/discopy/pull/767#discussion_r4040636782),
2026-09-17 19:23 UTC. The linter question is USER's to rule on. The second is answered by this
round: the six laws are all positive, stated on terms alpha-equivalent by construction, so an
`alpha_eq` that never says no passes every cell; the negative direction lives in unit tests only.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 19:31 `Canonical[C]`, a `Testable` tuple like `Renamed`: two shapes of one type, each generated
      under its own naming and under the one naming `"x"`, the second shape the first's or
      another; `TermBase.choices` shared with `shapes`
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 19:31 `alpha_completeness`: two terms are alpha-equivalent exactly when their canonical forms
      are equal, the direction the other laws lack; `assert_axioms` and the matrix cells green
      at both levels
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 19:31 the `proptest` label, so the matrix runs on CI rather than locally only
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 19:31 changelog, description, the thread answered
