# TODO

Prompt ([#429](https://github.com/discopy/discopy/issues/429), verbatim):

> Tracking a former `TODO` comment removed from `draw_controlled_gate` in #428.
>
> In `discopy/drawing/backend.py`, the nodes for the controlled part of a `Controlled` gate are built with `x=box.dom[0]`, i.e. the first wire of the whole box, regardless of where the controlled sub-gate actually sits:
>
> https://github.com/discopy/discopy/blob/add8d65c31018e92f1682927de0d4c92ad7d737f/discopy/drawing/backend.py#L373-L375
>
> The comment noted that the `x` coordinate should be selected properly for classical gates. Reproducing with a `Controlled` gate over classical wires should show the misplacement.
>
> The wire from the control dot to the target boundary is drawn with `bend_in=True, bend_out=True`:
>
> https://github.com/discopy/discopy/blob/add8d65c31018e92f1682927de0d4c92ad7d737f/discopy/drawing/backend.py#L444-L445
>
> The comment noted that `bend_in`/`bend_out` should be changed for the TikZ backend, whose bending presumably renders this wire differently from matplotlib. Comparing `draw(to_tikz=True)` output of a `Controlled` gate against the matplotlib output should show the discrepancy.

---

- [x] Reproduce the misplacement with a `Controlled` gate over classical wires
- [x] Select the controlled sub-gate's x-coordinate from the wire it actually sits on instead of `box.dom[0]`
- [x] Compare `draw(to_tikz=True)` against the matplotlib output and adjust `bend_in`/`bend_out` for the TikZ backend
- [x] Add regression tests and regenerate the affected docs images
- [x] Run `pflake8 discopy` and `coverage run -m pytest`
