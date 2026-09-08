# TODO

Review round on PR #658 by toumix, 2026-09-07 15:19 to 2026-09-08 07:04 UTC:

> I would rename the module to discopy.axiom, makes it sound more serious than testing.

> "carrier" isn't standard terminology for categories, not sure what it means in this case

> I don't understand this paragraph, surely we're not distinguishing alpha equivalent Python programs?
>
> ahh 😨 we are
>
> ok now i think i got a better understanding of what's going on here: either we're universally quantifying over a morphism of the category and the axiom is a method with self as first attribute, or we're quantifying over something else, most commonly an object of the category eg for identity, in which case we have a classmethod
>
> let's remove the hack and introduce two decorators axiom and classaxiom, which would be syntactic sugar for just axiom of a classmethod

> let's open an issue to add abc.DaggerCategory so this is not a
> needed anymore

> these attributes should be explained in the docs

> normal_form makes sense for Diagram not for MonoidalCategory in general

> there should be documentation for this attribute

> this is too hacky there must be another way
>
> I would suggest using two subclasses instead of distinguishing between the name of the arguments themselves

> shouldn't this class belong to the testing module?

> let's remove the non-alpha-invariant hack and replace it with two decorators axiom and classaxiom

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-08 07:12 `Axiom` is a law of an element and `ClassAxiom` a law of the carrier, told apart by the decorator — `axiom` and `classaxiom`, sugar for `axiom` of a `classmethod` — instead of the name of the first parameter; `abc.py` and `cat.py` declare their laws with `classaxiom`
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-08 07:12 Document every attribute of `Axiom`, say what a carrier is, and give `modulo` an example that makes sense of a `Diagram`
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-08 07:12 Rename `discopy.testing` to `discopy.axiom` — module, tests, docs page, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md` — and move `Equation` into it
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-08 07:12 Open an issue for `abc.DaggerCategory`, so a functor need not declare the dagger laws inapplicable
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-08 07:12 Run `pflake8` and the test suite, update the PR body, reply on and resolve every thread
