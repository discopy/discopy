# TODO

> Implement proposal B in https://github.com/rel-int/optyx/pull/16 as a
> comparison between three models: 1. A vanilla GNN, 2. a MapNN from
> discopy.neural, and 3. an optyx.interaction.CMap where nodes are
> interferometers with coherent memory and coherent messages

This branch hosts `discopy.neural` (the MapNN of #585, merged from its
fork branch on top of main) so that the optyx experiment can install it
from a discopy/discopy branch. The experiment itself lives on the optyx
branch of the same name; nothing here should merge except through #585.

- [x] Merge the head of #585 into this branch on top of main, resolving
      the `CHANGELOG.md` conflict by keeping both entries.
- [x] Run the three-model comparison against this branch: see
      `examples/beyond_1wl.{py,ipynb}` and `test/test_beyond_1wl.py` on
      the optyx branch `claude/proposal-b-three-models-awc5sc`. The
      MapNN separates the 1-WL-distinguishable control pair and is
      exactly constant on every 1-WL-equivalent pair, as the vanilla
      GNN is, while the photonic map separates the non-cospectral ones.
