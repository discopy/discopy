# Review round of 2026-09-17: USER on `alpha_eq_under`

> is this a joke? where are the substitutions?

— toumix, 2026-09-17 17:53 UTC, on `discopy/biclosed.py` line 642, the body of `TermBase.alpha_eq_under`, whose "substitutions" were dicts of binder depths.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 17:58 `alpha_eq_under` takes a `Substitution` for each term — `biclosed.Substitution`, the dataclass `closed.Substitution` now extends — of its bound variables by the fresh variable of their binder, the same one in every term, so that the terms are alpha-equivalent iff the substituted terms are equal, checked without building them; the fresh variables avoid the free names of the terms, collected once so the check stays linear; `TermBase.alpha_eq_under` abstract, `Constant`'s the equality of constants
