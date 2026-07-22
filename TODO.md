# TODO

Prompt (Alexis in the bridge, 2026-07-22, verbatim):

> yes let's go with rich display too, open a branch that targets Ale's if he opened one already

Implements [#425](https://github.com/discopy/discopy/issues/425) "Implement rich display hooks
for Diagram or Drawing" (author @0x0f0f0f), building on @0x0f0f0f's existing branch
`claude/ipython-discopy-rich-display-1o7he8` (commit ab7d6ac) which this branch targets.
Context: unblocks the Jupyter→marimo PR (#404) — marimo routes `plt.show()` to the console
channel, invisible in app preview/export; marimo scans `_repr_html_` → `_repr_mimebundle_` →
`_repr_svg_` → `_repr_json_` → `_repr_png_`; #404's per-notebook `show()` helper is temporary
and should become unnecessary.

---

- [x] Review ab7d6ac: what hooks exist, what's missing vs the marimo scan order —
  has `RichDisplay` mixin (`to_svg`, `_repr_svg_`, SVG-only `_repr_mimebundle_`),
  format/metadata threaded through the matplotlib backend, deterministic `svg_hashsalt`;
  missing: `image/png` in the mimebundle, figure-leak assertions
- [x] Merge origin/main into this branch (post-#421/#402 main; no force-push of Ale's branch) —
  one conflict in `backend.py` `Matplotlib.output`, resolved by combining main's
  reproducible-PNG metadata default with the branch's format/metadata threading + svg hashsalt
- [x] Implement `_repr_svg_` (and `_repr_mimebundle_` with svg+png) for `Diagram` and `Drawing` via the matplotlib backend, no display side effects —
  `RichDisplay.to_png` added, `_repr_mimebundle_` now returns both mimetypes
  lazily filtered by include/exclude; both svg and png byte-for-byte deterministic
- [x] Tests: hooks return valid SVG/mimebundle without opening figures; doctest examples —
  `test_rich_display` asserts svg header, png magic bytes, mimebundle
  include/exclude filtering, `plt.get_fignums() == []` and determinism for
  `Diagram`, `Drawing` and `Equation`; `RichDisplay` doctest covers the same
- [x] Check the #404 `show()` helper becomes unnecessary (a bare diagram as last cell expression renders) — note findings, don't touch #404's branch —
  simulated marimo's scan order (`_repr_html_` → `_repr_mimebundle_` → `_repr_svg_` →
  `_repr_json_` → `_repr_png_`) on `pregroup.Diagram`, `rigid.Diagram`, `Drawing` and
  `Equation`: all resolve at `_repr_mimebundle_` with svg+png, `plt.show` never called,
  `plt.get_fignums()` empty after; the helper (which returns `plt.gca()` and leaks an
  open figure) is now unnecessary — a bare diagram as last cell expression renders
- [x] Run `pflake8 discopy` and `coverage run -m pytest` — pflake8 clean;
  pytest 203 passed with quantum/tensor extras excluded (the sandbox proxy blocks
  the torch download so `uv sync --group all` fails; the 47 remaining failures are
  all `ModuleNotFoundError` for jax/sympy/pytket/pyzx/tensornetwork/nltk at import,
  unrelated to this change); `test_rich_display` and the `RichDisplay` doctest pass

---

Prompt (Alexis on PR #445, 2026-07-22, verbatim):

> should be rebased on the svg PR which should be rebased on main which already has the script for generating docs in CI

- [x] Rebase `claude/docs-png-to-svg-phs4aq` onto main and this branch onto it —
  svg branch rebased to `cb53f79` (tree byte-identical to its pre-rebase tip; one
  conflict: main moved `Diagram.transpose` from `rigid.py` to `abc.py`, kept the
  deletion); this branch linearized on top at `9ea8668`, one conflict in
  `Matplotlib.output` combining the svg branch's reproducible-metadata defaults
  with the rich-display `format`/`metadata` parameters
- [x] Fix the CI test (3.14) failure on `test_rich_display` — an earlier test in
  the full CI session leaves a figure open (only with all extras installed, e.g.
  the pyzx doctests), so `plt.get_fignums()` was `[1]` before the hooks ran;
  the test now calls `plt.close('all')` first so the assertion measures only
  the hooks; reproduced the failure mode and the fix locally
