# TODO

Prompt (USER on [#564](https://github.com/discopy/discopy/pull/564#discussion_r3806061086), verbatim):

> 2

i.e. the second of the two options offered on that thread:

> 2. **Real, separate issue** — give `frame_opening`/`frame_closing` the same
>    half-unit the bubble gets, so a square frame's wires are not squeezed.
>    That moves every `Equation` and frame baseline, so it wants its own PR.

Filed as [#597](https://github.com/discopy/discopy/issues/597).

---

- [x] File the issue with the measurement and the cause
- [x] ~~Give the frame boundary the bubble's full height~~ — superseded by
  USER's review below, which is what the branch does now
- [x] USER on [#598](https://github.com/discopy/discopy/pull/598#discussion_r3810719395),
  verbatim: *"We don't want to double the height of each bubble opening, would
  it help to introduce a box_height attribute the same way we have box_width?"*
  Introduced `box_height`, and split the boundary's layer asymmetrically —
  half a unit outside the frame, a quarter inside — so only the outer wires
  grow and the interior of every frame is left exactly as on `main`
- [x] Regenerate the baselines that move, and only those: 13 files, checked one
  by one against the failure list rather than with `OVERRIDE_DOCTEST_IMAGES`,
  which rewrites 40 images for their serialisation alone
- [x] A regression test measuring the half unit on a frame and on a bubble,
  checked to fail on `main` with a quarter
- [x] Eyeball `single-frame.svg` before and after
- [x] `uv run pflake8 discopy` and `uv run coverage run -m pytest --skip-extra`
- [x] `CHANGELOG.md` entry
