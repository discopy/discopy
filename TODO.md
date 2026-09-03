# TODO.md

Second review round from toumix on #658, 2026-09-03, on `discopy/testing.py`
as of dce7568, quoted verbatim.

> line 90 (`Natural.equation_factory`):
> that's a smell natural numbers should be defined in the main discopy
> code, arguably it's the most important concept of all mathematics ("God
> gave us the natural numbers") it feels wrong to downgrade it to a testing
> util

> line 330:
> I remember we wanted to make that wrapper into the main library so we
> avoid using the builtin int as objects of our category and remove the
> hacky sum in monoidal.Functor, let's file that as an issue if it's not
> already
>
> actually we already have that wrapper: it's called `PRO` for now but it
> would make much more sense to call it `Nat` then `PRO` is a
> `MonoidalCategory` with `Nat` as objects and we can even define `PROB` as
> a braided PRO, `PROP` as a symmetric `PROB`.

> line 343 (`TraceDinaturality.__new__`):
> this is hiding the fact that we sometimes want to treat natural numbers
> as sequences, i.e. unary encoding
> if that's what's happening here then we should go one step further and
> also implement slicing, iteration, etc.

> line 112 (`Atomic`):
> Same here, this is a foundational concept in universal algebra, not a
> testing utils. Let's treat it as such.

> line 145 (`Small`):
> "Subsingleton" seems to be a more standard way to call this

> line 162 (`BoundaryConnected`):
> This is a property of a morphism in a monoidal category, it would be
> nice if that was reflected in the type i.e. `value` cannot be an
> arbitrary `object`

> line 171 (`BoundaryConnected.__post_init__`):
> fishy, not sure what it would mean for a tuple to be boundary connected

> line 209 (`PastingDiagram.strategy`):
> why the local import?

> line 186 (`PastingDiagram`):
> This is a foundational concept of category theory, not a testing utils.
> I doubt that it's actually implementing a pasting diagram in the
> 2-categorical sense and even if it does it would be duplicate with just
> `Diagram`.

> line 208 (`PastingDiagram.strategy`):
> pasting diagrams need not be grids (i.e. each row may have different
> lengths)
>
> also this just seems to generate a `Diagram`, why call it `Pasting`?
>
> only reason I can see: this is a `Diagram` not with `Box`, but with
> `Diagram` as generators?

> line 242 (`ComposablePair`):
> one way to define this is as the image of a functor from the category 3
> with three objects {0, 1, 2} and two non-trivial arrows 0 to 1 and 1 to 2
>
> in general a pasting diagram as you defined above is just the image of a
> functor on a normal diagram

> line 249 (`ComposableTriple`):
> a.k.a. the number 4

> line 256 (`HorizontalPair`):
> a.k.a. ω the number 3 but rotated 90 degrees 😂

> line 262 (`Bifunctor`):
> noooo a bifunctor is a functor with a cartesian product as domain

> line 268 (`TraceSuperposing`):
> I would rather move this code to the traced module
>
> another high-level remark: shouldn't this be a subclass of Axiom?
> "superposing" isn't "a traceable arrow and an object to superpose" it's
> the equation that you get from those things as parameters

> line 334 (`TraceDinaturality` docstring):
> well no, in general the traced arrow shouldn't be traceable on its own if
> the sliding arrow isn't an endomorphism

> line 382 (`LeftCurrying`):
> Same here, I'd rather move this to biclosed

> line 412 (`FeedbackVanishing`):
> this feels duplicate from the axioms of traced, do we already have an
> issue to make traced a subclass of feedback?

> line 488 (`Relabelling`):
> smells like we're reinventing nominal set theory 😬
>
> apparently the standard name for this is "finite permutation"
>
> "relabelling" is fine too i think the name is more expressive

> line 495 (`Relabelling` docstring):
> it would feel cleaner if `Relabelling` had its own `then` method?

> line 515 (`Relabelling.__getitem__`):
> we're mixing layers of abstractions here, this should be handled by
> rigid.Functor already it feels wrong to do it again here

> line 540 (`Relabelled`):
> this goes away, it's way too hacky; why not just add a case in the
> getitem of Relabelling?

> line 577 (`Axiom`):
> I would have expected this to be the first class of the overall file,
> taking a break before diving deeper

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 `Axiom`, `AxiomFailure` and `axiom` open the module, before `Strategy` and the wrappers.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 `Small` becomes `Subsingleton`; `Bifunctor` becomes `Square`, a bifunctor being a functor out of a product.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 The wrappers' `value` fields are typed `C0` and `C1`, and `BoundaryConnected` checks the cells of a `PastingDiagram` rather than of any tuple.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 `TraceDinaturality`'s docstring says the arrow is traceable only once the sliding arrow is composed in.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 `Relabelling` looks atoms up by name only, leaving rotations to `rigid.Functor` and delays to `feedback.Functor`; `Relabelled` is a case of its `__getitem__`, so a generated functor is `cls(labelling, labelling)`.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 Answer the `then` question: a `then` on `Relabelling` only helps once `MappingOrCallable.then` delegates to it, which is the composition redesign #648 asks for.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 Answer the local import: `hypothesis` is a test dependency and `discopy.abc` imports this module, so it must import without it.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 File the issue for `Nat` in the main library — `PRO` renamed, `testing.Natural` retired, the unary sequence protocol, `PROB`/`PROP` — and answer the three threads with it.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 File the issue for traced categories as feedback categories, since none exists, and answer the `FeedbackVanishing` thread with it.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 File the issue for the argument wrappers as objects of the main modules — pasting diagrams as images of functors from finite categories, trace and currying arguments beside `traced` and `biclosed`, `Atomic` as a generator — with the import cycle it must solve, and answer the eight threads with it.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 14:05 Keep the name `Relabelling`, as the thread concludes, and resolve it.
