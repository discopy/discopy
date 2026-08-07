# TODO

> Task: give feedback loops a directed drawing in DisCoPy.
> Right now a feedback loop and a compact-closed trace draw identically, so a picture cannot tell you which one you are looking at. `feedback.Feedback.to_drawing` (`discopy/feedback.py:555`) is literally `self.arg.to_drawing().trace()`, and `Drawing.trace` (`discopy/drawing/drawing.py:679`) builds the loop from two boxes:
>
> ```
> cup = Box('cup', cup_dom, Ty(), draw_as_wires=True).to_drawing()
> ```
>
> `cap = Box('cap', Ty(), cap_cod, draw_as_wires=True).to_drawing()`
> Both are `draw_as_wires=True`, so they render as plain bent wires. The loop should instead carry an arrow pointing backwards along it, showing that the memory flows from the codomain back to the domain one time step later. More generally, DisCoPy has no way to draw a directed edge at all, and this is the motivating case for one.
> What I have already checked, so you don't have to:
>
> * `BOX_DRAWING_ATTRIBUTES` is a `name -> lambda box: default` dict in `discopy/config.py:15` (`drawing/drawing.py` only imports it). It has `draw_as_braid`, `draw_as_discards`, `draw_as_measures`, `draw_as_controlled` — nothing directional.
> * Arrowed wires already render: `draw_wire(..., style='->')` at `drawing/backend.py:436`, but it is reachable only from inside `draw_measure`.
> * All three backends already accept the parameter — `Backend` (`backend.py:102`), `TikZ` (592), `Matplotlib` (781) each define `draw_wire(self, source, target, bend_out=False, bend_in=False, style=None, linewidth=None)`.
> * The gap is `Backend.draw_wires` (`backend.py:300`): it calls `self.draw_wire(source_position, target_position, bend_out, bend_in, linewidth=...)` and never passes `style`. Cups and caps go through this path because they are `draw_as_wires`.
>
> Suggested shape, but treat it as a starting point rather than a specification — you have the code in front of you and I don't:
>
> 1. Add `"draw_as_feedback": lambda _: False` to `BOX_DRAWING_ATTRIBUTES`.
> 2. Let `Drawing.trace` mark its cup and cap with it — a keyword such as `trace(..., feedback=False)`, or a sibling method, whichever reads better against the rest of that module.
> 3. In `draw_wires`, pass `style='->'` when an endpoint is a box node whose box has `draw_as_feedback`.
> 4. Make TikZ emit the matching `->` edge option so documentation builds agree with matplotlib.
> 5. Point `feedback.Feedback.to_drawing` at the new path.
>
> Open questions worth deciding deliberately: which way the arrow should point and where it should sit on the loop (mid-wire, or at the cap); whether `Drawing.trace` should grow a flag or whether feedback deserves its own drawing method, given that a trace and a feedback loop are genuinely different operations; and whether the attribute should be general (`directed`) rather than feedback-specific, since directed edges are useful beyond this case.
> Please open an issue describing the problem, then a PR with the change, a test or doctest, and a drawn example in the docs. The downstream consumer is optyx, where every stateful diagram carries one of these loops, context in [rel-int/optyx#12, discussion_r3691951535](https://github.com/rel-int/optyx/pull/12#discussion_r3691951535). Note that thread says `BOX_DRAWING_ATTRIBUTES` is in `drawing/drawing.py`; that is wrong, it is `config.py`.

- [x] Open the issue describing the problem ([#515](https://github.com/discopy/discopy/issues/515))
- [x] Add `draw_as_feedback` to `BOX_DRAWING_ATTRIBUTES`
- [x] Give `Drawing.trace` a `feedback` keyword marking its cup and cap
- [x] Draw the arrow in `Backend.draw_wires`, on the wire from the cap to the cup
- [x] Implement `draw_arrowhead` in the Matplotlib and TikZ backends
- [x] Point `feedback.Feedback.to_drawing` at the new path
- [x] Fix the missing comma between a TikZ wire style and its looseness
- [x] Add tests and a drawn example in the docs, regenerate the feedback baselines
- [x] Run `pflake8 discopy` and the test suite

> Is the method tested on diagrams with more than one feedback loop?

- [x] Test diagrams with more than one feedback loop
