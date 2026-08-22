# TODO

Three bugs found while implementing #370, each with a ruling from Alexis on its issue.

[#492](https://github.com/discopy/discopy/issues/492) — `closed.Substitution` drops constants,
loses the `left` flag and recurses forever on abstractions:

> open a small PR to fix this

[#493](https://github.com/discopy/discopy/issues/493) — `python.Function.tensor` is binary where
the categorical interface is variadic:

> just add @unbiased and this is fixed
>
> a bigger problem is that we don't have nullary tensor defined, we need Monoid.unit().tensor(*args)

[#494](https://github.com/discopy/discopy/issues/494) — `biclosed.Constant` documents an `inside`
attribute that does not exist:

> good catch! open a PR

- [x] `closed.Substitution` returns constants unchanged, keeps `left` and recurses into the body
- [x] `python.Function.tensor` is `@unbiased` in `multiplicative`, `additive` and `finset`
- [x] `biclosed.Constant` documents the attributes it has
- [x] Tests for each of the three, `CHANGELOG.md` entry, `pflake8` and `pytest` green

`python.Function.then` was binary for the same reason and got the same decorator: `cat.Arrow.then`
is variadic and generic code calls it unbound just like `tensor`.

- [x] Answer cubic's review of 2026-08-16: two findings acted on, two not

Two of cubic's four findings were real:

- `@unbiased` swallowed a first argument given by keyword into `**params` and returned `self`
  unchanged, so `f.then(other=g)` silently composed nothing. That was a regression for
  `python.Function`, whose biased methods took `other=`, and a latent one for the thirteen other
  `@unbiased` methods. Fixed in `unbiased` itself, with a test.
- the `Application` assertion in `test_Substitution` only checked `.left`, which the input already
  had. It now asserts the whole substituted term.

The other two were not:

- the doctest ending in `\\` is a single backslash once Python parses the docstring, i.e. a valid
  continuation. `pytest --doctest-modules discopy/closed.py` passes, and it passed on every CI run
  of this branch. The suggested one-line replacement would be 81 characters and fail `pflake8`.
- variable capture is real but deliberate here: it is #510, and renaming needs a ruling on where
  fresh names come from. Fixing it inside this PR is the scope creep the issue exists to avoid.

Filed rather than fixed here:

- [#509](https://github.com/discopy/discopy/issues/509) nullary `tensor`, i.e. the second half of
  Alexis' comment on #493: `@unbiased` folds from `self`, so the empty product needs a `unit` on
  the category, which changes `abc.MonoidalCategory`.
- [#510](https://github.com/discopy/discopy/issues/510) `Substitution` captures free variables:
  the bound variable is never substituted but it is never renamed either, and renaming needs a
  source of fresh names that keeps `repr` deterministic.
