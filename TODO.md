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

- [ ] Review ab7d6ac: what hooks exist, what's missing vs the marimo scan order
- [ ] Merge origin/main into this branch (post-#421/#402 main; no force-push of Ale's branch)
- [ ] Implement `_repr_svg_` (and `_repr_mimebundle_` with svg+png) for `Diagram` and `Drawing` via the matplotlib backend, no display side effects
- [ ] Tests: hooks return valid SVG/mimebundle without opening figures; doctest examples
- [ ] Check the #404 `show()` helper becomes unnecessary (a bare diagram as last cell expression renders) — note findings, don't touch #404's branch
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`
