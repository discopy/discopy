# TODO — issue #427: Fix unrolled drawing in feedback docstring

> The above image results from the module-level docstring of the
> `discopy.feedback` module. The offending PR is #421, and specifically the
> offending line is the following: discopy/feedback.py#L67. #421 changed the
> behavior of `.name` in a way that broke this test.

Approved by Alexis: P7 go (bridge, 2026-07-22)

- [x] Diagnose exactly what #421 changed about `Ty.name` / `Ob.name` —
  `monoidal.Ty.__init__` now sets `name` to `type(self).__name__` (e.g. "Ty")
  instead of `str(self)`, so `stream.Ty.sequence(x.name)` iterated the string
  "Ty" character-wise into `T`/`y` wire pairs
- [x] Fix the feedback module so the docstring image renders correctly
  (use `x.inside[0].name`, the idiom of `rigid.py` and `grammar/categorial.py`,
  on lines 67 and 118)
- [x] Regenerate `docs/_static/feedback/*.png` and commit the corrected image
  (only `slide-unroll.png` changed; it matches the pre-#421 drawing)
- [x] Add a regression test: a doctest assertion pinning the unrolled wire
  names `x0 @ x1 @ x2` next to the image, which fails with the old `.name`
- [x] Check whether other drawings / docstrings were affected — grep found no
  other `.name`-on-`Ty` usage; only `slide-unroll.png` blew up when CI
  regenerated `docs/_static` after #421
- [x] Run `uv run pflake8 discopy` (clean) and `uv run coverage run -m pytest`
  (feedback, monoidal, stream and drawing all pass; 45 pre-existing failures
  from missing optional dependencies — sympy, jax, torch — unchanged with or
  without the fix, torch install blocked by the network proxy)
