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
- [x] the matrix on every level: each failure classified, `.modulo` a quotient where the free category identifies terms, `.failing` with a reason where a law breaks, `.inapplicable` where a structure is absent
- [x] docs autosummary, CHANGELOG entry, `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark` green

## Round: degenerate shapes

> do we still need shapes like TraceVanishing or TraceSuperposing? arent hints doing a similar thing?
>
> do it

- [x] `TraceSuperposing` and the currying shapes draw a real arrow through the search rather than an identity or the evaluation itself; `FeedbackVanishing` and `HomogeneousMemory` go, the vanishing law taking an arrow; the matrix reclassified where the wider reach changes a verdict; CHANGELOG

## Round: the sequent-pattern language

> is there any way to unify axioms and rules? don't implement anything just think about it
>
> do level 2, the sequent-pattern language

- [x] `Var`, `Pattern` and sequent patterns in `discopy/axioms.py`: matching with backtracking and inversion of adjoints, delays and exponentials, instantiation by kind; tests
- [x] `@leaf` and `@rule` from patterns, with the hint derived from the conclusion pattern; the structural rules of `discopy.abc` restated as patterns, hand-written hints gone; `search` checks the sequent a rule claims
- [x] `Shape` with declared premises and returns, validator and strategy generic; the arrow shapes restated as patterns, `Grid` and the bespoke shape classes gone
- [x] the matrix green, docs autosummary, CHANGELOG, `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark`

## Round: sequents from annotations

> extract the sequent pattern from the type signature of axioms, we shouldn't need any shape anymore. everything should be constructed from annotations

- [x] `Axiom.strategy` draws every argument from its annotation: `C1[A, B]` an arrow of that sequent, a `Var` an object of its kind, shared metavariables drawn once; `weaken` takes a predicate on the equation; tests
- [x] every axiom of `discopy.abc` annotated with its sequents, `Shape`, `Atomic`, `NonEmpty` and `BoundaryConnected` gone, the connected subspace a predicate; module declarations and tests follow
- [x] the matrix green, docs, CHANGELOG, `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark`

## Round: efficiency of the annotations

> is this still as efficient as the previous shape implementation?

- [x] measured: `bifunctoriality` 15.7 → 46 ms per example, the predicate rejecting whole tuples the wrappers filtered arrow by arrow; a `Subspace` carries the generation parameters, `connected` draws no closed component, 13.2 ms after

## Round: metavariables as type parameters

> instead of defining a bunch of variables at the top of abc, define variables using generic syntax like
> ```
>     @axiom
>     def composition_cod_typing[A: C0, B: C0, C: C0](cls, f: C1[A, B], g: C1[B, C]) -> ...
> # and
>     @axiom
>     def feedback_joining[A: C0, M: C0](cls, f: C1[A @ M.d, A @ M], mem: M) -> ...
> ```

- [x] a law's metavariables are its type parameters, their kind the bound: `C0` a type, `Atom[C0]`, `NonEmpty[C0]`, `Pair[C0]`; `M.d` is the delay; rules read their conclusion and premises off annotated `dom`, `cod` and premise parameters the same way; tests
- [x] every axiom and rule of `discopy.abc` declared with type parameters, the module-level metavariables gone
- [x] the matrix green, docs, CHANGELOG, `uv run pflake8 discopy`, `uv run coverage run -m pytest`, `uv run pytest proptest/ -n auto -p no:benchmark`

## Round: the fast profile and the pattern EDSL

> * the full property testing suite is too expensive to run every time, create a new hypothesis profile that has the same hypothesis settings as the default except that when enabled, every axiom is tested once bound to its defining class, and avoid re-testing inherited axioms. this is the first thing you should do and from now you should use this fast profile.
> * consolidate Pattern as a strongly typed edsl with a base class PatternBase + concrete dataclass + closed type alias Pattern[T] as a union
> * it would make more sense if Axiom[P, T].strategy returned a st.SearchStrategy[Equation[T]], and instead have the input strategy stored as self.pattern.strategy. make Axiom and Pattern inherit from Testable

- [x] the `fast` profile in `proptest/conftest.py`, the `dev` settings, under which the matrix keeps one cell per declaration of a law: the enrolled type nearest the class declaring it; `CONTRIBUTING.md`; the broken cells checked without shrinking
- [x] `PatternBase` and the concrete dataclasses `Var`, `Adjoint`, `Delay`, `Exp`, `Word`, `Alternatives`, `Sequent`, `Hom`, `Signature`, with `Pattern[T]` the closed union; `Op`, `pattern`, `sequent`, `sequents`, `matching`, `resolve` gone; tests
- [x] `Axiom` and `PatternBase` subclass `Testable`; `Axiom.pattern` is the `Signature` of its annotations and `Axiom.strategy` maps its strategy to equations; `falsify` returns the equation; the matrix and `assert_axioms` follow
- [x] docs, CHANGELOG, `uv run pflake8 discopy`, `uv run pytest`, `HYPOTHESIS_PROFILE=fast uv run pytest proptest/ -n auto -p no:benchmark`

## Round: the canonical equation

> define Axiom.canonical which instantiates the pattern with the default names and returns the axiom schema as an equation over diagrams, then Axiom.draw which draws this canonical equation

- [x] `PatternBase.canonical`: each metavariable an object named after it, each sequent a box named after its parameter; `Axiom.canonical` the law on those, `Axiom.draw` drawing it; tests, a drawn baseline, CHANGELOG, lint, `uv run pytest test/axioms.py`, the fast matrix

## Round: rules on the methods

> remove rules and leaves and instead directly apply @rule or @leaf to the methods like feedback cups caps etc... no feeding_back cupping capping

- [x] `@leaf`/`@rule` mark the structural method itself: the sequent it concludes read off its return annotation `-> C1[dom, cod]`, its premises off the parameters annotated `C1[..]`, `self` included, its other arguments off their patterns; `Count` and `Bool` kinds with `X ** N` and `L[a, b]` for the legs of a spider, the copies and the side of a trace or an evaluation; `inapplicable(reason)` on an override; `Category.rules` reads the rule a method carries through plain overrides; the named rules gone; tests, docs, CHANGELOG, lint, `uv run pytest`, the fast matrix

## Round: two-categories

> * everything in abc should be either a rule or a generator (what you called leaf) or an axiom
> * axioms and rules should directly store a pattern
> * patterns should automatically typecheck inputs and output upon call
> * patterns should work for 2-categorical shapes as well. MonoidalCategory is actually a 2-category where 0-cells are colours, wires are 1-cells and boxes are 2-cells. I've started splitting a new TwoCategory class in abc, it should define all the axioms and rule, while MonoidalCategory should simply inherit them naturally. for now i've only covered tensor and bifunctoriality, do the rest.

- [x] levels: `C0`, `C1`, `C2` are `Level` patterns, `C1[R, G]` a sequent of 1-cells, prepared into every class body by the metaclass of `Testable` so that a type parameter's bound may name them; a metavariable's kind is its bound, `Atom[C1[R, G]]` included, and a variable bounded by a sequent matches and draws its boundaries; `Cells(category, top)` resolves a level to the class of its cells, the top being the declaring class's last type parameter
- [x] `Axiom` stores its `pattern` and `result` at decoration and its `owner` at `__set_name__`; calling it typechecks the arguments against the pattern and the terms of the equation against the result
- [x] `Rule` stores its `conclusion` and `pattern`; one derivation for every rule: a feasible match drawn, an identity premise where the match allows, the unbound type variables drawn from the types and the hints, the proofs and the built cell typechecked; `id`, `then` and `tensor` declarative, `box` and `permutation` procedural; `identity`, `cut`, `tensoring`, `splits`, `atoms` gone, `hints` a function
- [x] `TwoCategory` states the monoidal axioms and rules on 0-, 1- and 2-cells, `MonoidalCategory` inherits them; `generator` the name of `leaf`; tests, docs, CHANGELOG, lint, `uv run pytest`, the fast matrix

## Round: real colours

> make MonoidalCategory subclass TwoCategory[Colour, C0, C1] with real colours

- [x] `abc.Colour`, the 0-cells, `MonoidalCategory(TwoCategory[Colour, C0, C1])` resolving to the category's own colours; a level substituted by a superclass of the positional cells keeps the cells; `BraidedCategory` and `RigidCategory` declare the one trivial colour crossing and bending wires need; tests, CHANGELOG, lint, `uv run pytest`, the matrices

## Round: generators

> these solvers aren't good, lets return to the initial approach of defining rules, generators and axioms and deriving search strategies from there.
> allow diagrams to adjust the set of generators that the strategy can invoke. just like axioms collect all axioms, generators should be a similar method that can be overridden. use this to generate random sentences for pregroup or generate quantum circuits.

- [WIP] @af1f3450-2026-09-13 19:10 `Category.generators`, the rules building a box from nothing, collected like `axioms` and overridable by a class attribute; `search` invokes the structure of `rules` and the generators of `generators`; `Rule.constant(box)` the generator of one given box; random sentences of a pregroup grammar and random quantum circuits from a gate set, in tests and doctests; CHANGELOG, lint, `uv run pytest`, the fast matrix
