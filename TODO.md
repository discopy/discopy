# TODO

USER, 2026-09-16 11:26 UTC, two review comments on `discopy/closed.py` of
https://github.com/discopy/discopy/pull/400 at `4668990f`:

On `TermBase.alpha_key` and `alpha_equivalent`:

> let's remove this from the PR for now, we have open PRs with a faithful round trip from term to diagram

On `TermBase.normal_form`:

> same let's remove this, it doesn't have much to do with ACG

- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 12:20 `alpha_key`, `alpha_equivalent`, `normal_form`, `weak_head_normal_form` and the `Substitution` machinery behind them are out of the PR; what `is_linear` and `compose` still need stays.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 12:20 The docs, tests and changelog check terms by evaluating them rather than by normal forms.
