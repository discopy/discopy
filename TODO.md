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
- [x] ~~Build it~~ — **reverted, the branch is back to zero diff.** Three
  attempts, all wrong for the same measured reason: the space they added came
  out of the frame's own border, which on `main` is a uniform **0.5 all
  round**, so the top and bottom bands grew to **0.747** while the sides
  stayed **0.5** — USER on [#598](https://github.com/discopy/discopy/pull/598#discussion_r3811580106),
  verbatim: *"there should not be a diff here. it breaks the frame drawing
  which doesn't have the same width on top and sides anymore"*
- [x] Measure `main` rather than argue about it: a frame there is **0.25 of
  wire outside on every side and a 0.5 band on every side**, i.e. already
  uniform. #597 compared the top wire against a *bubble's* half unit, which is
  half an opening curve and not a wire stub at all. The premise was wrong
- [x] USER on [#598](https://github.com/discopy/discopy/pull/598#discussion_r3811589380),
  verbatim: *"you've been fiddling with margins but you haven't fixed the
  actual problem: the bubble is broken here"* — reproduced on
  `bubble-example.svg`: the inner square bubble draws no outline at all. That
  is [#520](https://github.com/discopy/discopy/issues/520), and
  [#564](https://github.com/discopy/discopy/pull/564) fixes it. Rendered on
  #564's branch the same picture comes out with its outline
- [ ] **Blocked on a ruling.** Recommendation: close #597 as fixed by #564 and
  close this PR. If instead the outside wire should still grow, it has to be
  added by `Drawing.bubble` outside the frame rather than inside its boundary
  layer, so that the band stays 0.5 — and it will then be 0.5 on top and
  bottom against 0.25 on the sides, which is the same complaint one level out
