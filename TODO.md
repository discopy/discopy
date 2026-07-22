# TODO — issue #427: Fix unrolled drawing in feedback docstring

> The above image results from the module-level docstring of the
> `discopy.feedback` module. The offending PR is #421, and specifically the
> offending line is the following: discopy/feedback.py#L67. #421 changed the
> behavior of `.name` in a way that broke this test.

Approved by Alexis: P7 go (bridge, 2026-07-22)

- [ ] Diagnose exactly what #421 changed about `Ty.name` / `Ob.name`
- [ ] Fix the feedback module so the docstring image renders correctly
- [ ] Regenerate `docs/_static/feedback/*.png` and commit the corrected image
- [ ] Add a regression test if feasible
- [ ] Check whether other drawings / docstrings were affected by the same change
- [ ] Run `uv run pflake8 discopy` and `uv run coverage run -m pytest`
