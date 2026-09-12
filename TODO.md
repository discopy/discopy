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
- [x] `Ty`, `Wire` and `Box` strategies in `monoidal` and the levels whose wires carry more (`rigid`, `frobenius`, `feedback`); `atoms`, `splits`, `tensoring` on `MonoidalCategory`; `monoidal.Diagram.strategy` on the search, enrolling every diagram level
- [x] structural leaves and rules per level with hints: braiding, twisting, permuting, cupping, capping, evaluating, copying, spidering, tracing, feeding_back; `assert_strategy_finds` and a `test_strategy` per module pinning reach inside a constrained hole
- [x] the matrix on every enrolled level: classify each failure per the protocol, `.failing` where a bug is found, fix where it is the strategy's
- [x] `conftest.py` drops the `filter_too_much` suppression; module docstring section, bibliography, CHANGELOG entry
- [x] `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark` green

Rough edges met on the way, each worth an issue:

- [ ] file: `feedback.Diagram.trace` is inherited from `markov` and returns a `markov.Trace` that is not a feedback diagram, so it cannot be composed; declared `tracing` inapplicable on `feedback.Diagram` meanwhile
- [ ] file: `frobenius.Wire` and `feedback.Wire` carry no colours, `frobenius.Wire.l` returning itself and `feedback.Wire.__init__` taking none, so their types are generated transparent; a coloured frobenius type breaks `dagger_contravariance`
- [ ] file: the roundtrip laws declared failing under #742 on `traced`, `biclosed`, `markov`, `ribbon`, `frobenius` and `feedback` diagrams each name the box whose `serialised_attrs` are missing: `Trace`, `Eval`, `Copy`, `Twist`, `Spider`, `Feedback`

## Round: the abc axioms

> reintroduce all abc axioms of all categories, introducing the necessary shapes and making use of that new architecture

- [x] the argument shapes as `Testable` wrappers in `discopy/axioms.py`, generating through the constrained search: `Atomic`, `NonEmpty`, `HorizontalPair`, `Square`, the trace and feedback shapes, the currying shapes; tests in `test/axioms.py`
- [x] the axioms of every level of `discopy.abc` restated beside the rules, from `bifunctoriality` to `reidemeister_1_cup`
- [WIP] @claude-3f440127-2026-09-12 13:55 the matrix on every level: each failure classified, `.modulo` a quotient where the free category identifies terms, `.failing` with a reason where a law breaks, `.inapplicable` where a structure is absent
- [ ] docs autosummary, CHANGELOG entry, `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark` green
