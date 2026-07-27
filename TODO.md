we need to resolve the conflicts, and the comment by toumix

https://github.com/discopy/discopy/pull/445

do we need format? I feel like that could be an attribute of the backend itself?
what is metadata used for? if it's just to remove them I would just remove them by default

what the fuck is going on ? explain and evaluate all the possible solutions with their tradeoffs

- [x] Merge `origin/main` into the branch, resolving the `draw` conflicts by
  dropping the `metadata` parameter and making `format` an attribute of the
  `Matplotlib` backend, as suggested in the review.

- [WIP] @01H4qh8wP78PLoRLjhd6NwyK-2026-07-27 14:18 Reply to toumix's comment on #445.

Mathematical design: saving a figure is a function of the format, determined
by the path when it is a file name and given explicitly when it is an
in-memory buffer, so the format belongs to the Matplotlib backend that does
the saving. The metadata is not a parameter: reproducible output strips it
unconditionally.

Verification: `pflake8 discopy` passes and `pytest` gives 733 passed with the
five `biclosed`, `cmap` and `test/cmap.py` failures reproducing identically on
`origin/main` (the Graphviz `dot` executable is missing from this
environment).
