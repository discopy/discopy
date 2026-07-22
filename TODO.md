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
- [WIP] @bridge-2026-07-22-fbdoc Fix the feedback module so the docstring image
  renders correctly (use `x.inside[0].name`, the idiom of `rigid.py` and
  `grammar/categorial.py`)
- [WIP] @bridge-2026-07-22-fbdoc Regenerate `docs/_static/feedback/*.png` and
  commit the corrected image
- [ ] Add a regression test if feasible
- [ ] Check whether other drawings / docstrings were affected by the same change
- [ ] Run `uv run pflake8 discopy` and `uv run coverage run -m pytest`
