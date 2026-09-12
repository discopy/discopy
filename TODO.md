# TODO

> this conversation is about defining a search strategies for monoidal diagrams, cmaps, hypergraphs, tensor terms, networks, grammar to implement property testing in discopy.
> take a step back from the existing implementation and think about how to make better search strategies over this huge variety of diagrams, both syntactic categories (diagram, term) and semantic categories (matrix, tensor, grammar, quantum), but also cmap and hypergraph. One issue i've had in the design of this property testing suite is that while ive tried to make is as little constrained as possible, some shapes (such as the ones for trace, or Sum although we've never implemented that one) require constraining both the dom and the cod of a diagram. generating a diagram with both boundaries constrained can only fill in immediately with a trivial box to close a hole, otherwise it is computationally hard as it corresponds in non-free categories to a word problem, but i am afraid it might be not powerful enough to generate a large enough subspace of all diagrams, on the other hand it is also penalized to overshoot and generate invalid diagrams and post-filtering, as it wastes a lot of generation time. what if we took the naming "search strategies" literally and implemented diagram generators as proof search? can you implement by or indirectly compile to ? would it help with hypothesis statistics? could focusing help us here? analyze the literature about word problems and type-directed term synthesis in such categories and come up with a way to realize search strategies as efficient solvers.
>
> rebase on the refactor/serde branch, which is the new simplified property testing infrastructure

Decisions taken in session: rules live on `discopy.abc` beside the axioms, one generic search
interprets them; hand-rolled in `st.composite`, no new dependency; re-landed on a branch cut
from `refactor/serde` since the four commits on `split/2` touch code that base does not have;
monoidal diagrams enrol in the matrix on this branch, every level through inheritance.

- [x] `Rule`, `@leaf`, `@rule` and `search` in `discopy/axioms.py`; `Testable.declarations` shared by `axioms` and `Category.rules`; `identity`, `box`, `cut` on `Category`; `cat.Arrow.strategy` on the search
- [WIP] @claude-3f440127-2026-09-12 12:34 `Ty`, `Wire` and `Box` strategies in `monoidal` and the levels whose wires carry more (`rigid`, `frobenius`, `feedback`); `atoms`, `splits`, `tensoring` on `MonoidalCategory`; `monoidal.Diagram.strategy` on the search, enrolling every diagram level
- [ ] structural leaves and rules per level with hints: braiding, twisting, permuting, cupping, capping, evaluating, copying, spidering, tracing, feeding_back; `assert_strategy_finds` and a `test_strategy` per module pinning reach inside a constrained hole
- [ ] the matrix on every enrolled level: classify each failure per the protocol, `.failing` where a bug is found, fix where it is the strategy's
- [ ] `conftest.py` drops the `filter_too_much` suppression; module docstring section, bibliography, CHANGELOG entry
- [ ] `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark` green
