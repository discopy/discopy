transparent backgrounds go brrr
discopy's readme is gonna look as cool on black background!
make sure you find a trick to have the same image draw on both white and black, i.e. add white borders to every black wire
[https://github.com/discopy/discopy/issues/453](https://github.com/discopy/discopy/issues/453)

- [x] Make drawing backgrounds transparent, boxes white by default, and black wires visible on light and dark backgrounds.
- [x] Add focused tests and regenerate affected documentation images.
- [x] Run lint and the full test suite.

## Review feedback

- [x] not sure where it comes from but let's add an option for the border colour too (with transparent as default)
- [x] artefact of drawing equation symbols as white spiders, let's remove it
- [x] something's going very wrong here
- [x] background inside the frames should be transparent too!

## Review feedback, second round (2026-08-03)

- [x] wires should have no outline on coloured regions, only when the region colour has a non-FF alpha; outline anticorrelated to region transparency, half-outlines if possible (single-frame, coloured-box, coloured-frame threads)
- [x] spider nodes should have an outline too when the background is transparent (feedback-random-walk thread)
- [x] coloured-bubble corners are funky: diagnose and fix
- [x] ribbon dual_rail messed up: region-drawing bug introduced or made visible? diagnose and answer
- [x] monoidal bubble-example looks weird even in the base branch: diagnose, file issue if pre-existing (filed [#520](https://github.com/discopy/discopy/issues/520))

## Review feedback, third round (2026-08-04)

- [WIP] @claude-2026-08-04 18:30 the frame is completely invisible in single-frame: draw the frame contour underneath the region fills, keeping the interior transparent
- [WIP] @claude-2026-08-04 18:30 wire labels and bare text (equation symbols) unreadable on black: white halo, like subtitles
- [ ] wires look doubled on black backgrounds; prefers-color-scheme CSS proposed as an alternative to outlines — design ruling for a human, answered on the thread
