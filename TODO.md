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

- [WIP] @claude-br96gc-2026-08-03 15:35 wires should have no outline on coloured regions, only when the region colour has a non-FF alpha; outline anticorrelated to region transparency, half-outlines if possible (single-frame, coloured-box, coloured-frame threads)
- [WIP] @claude-br96gc-2026-08-03 15:35 spider nodes should have an outline too when the background is transparent (feedback-random-walk thread)
- [WIP] @claude-br96gc-2026-08-03 15:35 coloured-bubble corners are funky: diagnose and fix
- [WIP] @claude-br96gc-2026-08-03 15:35 ribbon dual_rail messed up: region-drawing bug introduced or made visible? diagnose and answer
- [WIP] @claude-br96gc-2026-08-03 15:35 monoidal bubble-example looks weird even in the base branch: diagnose, file issue if pre-existing
