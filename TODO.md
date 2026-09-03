# TODO.md

Review round on #658, 2026-09-03: six threads from toumix and three from
CodeRabbit on dce7568, quoted verbatim.

> toumix, `.github/workflows/build.yml` line 61:
> workflow code should speak for itself, I don't think we should add
> comments here
>
> another point: isn't proptest its own workflow? if it runs in parallel I
> don't see why we're increasing the budget for this one?

> toumix, `AGENTS.md` line 11:
> this @ syntax is interpreted by Claude Code as inlining the prompts, we
> should replace it with links (for the other bullet points) and only use @
> in Claude.md (other harnesses don't follow the same convention)
>
> in fact, i'm not sure we need to include 200 lines of prompt about
> property testing in each agent context, it should only be needed when
> things go wrong?

> toumix, `PROPTEST.md`:
> I'm not sure this is specific to agents; humans could develop against PBT
> too
>
> Also I feel like this should go as documentation of discopy.testing rather
> than a standalone markdown

> toumix, `PROPTEST.md` line 15:
> I don't see why we have a special case here, maybe this could be
> documented the same way as axioms?

> toumix, `PROPTEST.md` line 47:
> ha is that the difference between axioms and the other properties? do we
> really need equations?

> toumix, `PROPTEST.md` line 77:
> ha now it's clearer why you want Equation rather than bool, but why
> restrict it to the category-theoretic axioms? e.g. we could have
> `Equation(eval(repr(x)), x)` too

> CodeRabbit, `.github/workflows/proptest.yml` line 40:
> **Do not persist the job token in the checkout.** The labelled
> `pull_request` job executes pull-request-controlled tests.
> `actions/checkout@v7.0.1` persists the read token in local Git
> configuration by default. Set `persist-credentials: false`.

> CodeRabbit, `discopy/testing.py` line 836:
> **Handle `NotImplemented` in the dry run.** When a parameterised axiom
> returns `NotImplemented`, `assert axiom(*args), axiom` converts it to a
> Boolean. Python 3.9–3.13 treats it as true with a `DeprecationWarning`;
> Python 3.14 raises `TypeError`. Store the verdict and accept
> `NotImplemented` explicitly, as `falsify` does.

> CodeRabbit, `proptest/carriers.py` line 14:
> **Enroll diagram carriers before claiming diagram coverage.** `cat.Arrow`
> is the base class of `monoidal.Diagram`, not a subclass. `cat.Functor` is
> not a diagram class. Therefore `DIAGRAMS` is empty, and pytest skips the
> normal-form, foliation, and drawing properties. Add diagram carriers with
> working strategies, or defer the coverage claims until collection
> produces nonempty diagram cells.

> CodeRabbit, pre-merge check:
> Docstring coverage is 55.63% which is insufficient. The required threshold
> is 80.00%. Docstring coverage is scoped to functions touched by this diff.

- [x] `build.yml`: drop the comment and the timeout bump, every run of this branch since the bump finishes in about five minutes.
- [x] `AGENTS.md`: links in place of the `@` imports, and the property-testing pointer out of the must-read list, to be read when a property fails or a carrier or a law is added.
- [x] `PROPTEST.md` becomes the documentation of `discopy.testing`: its content moves into the module docstring in a voice for every developer, `discopy.testing` joins the API docs, every reference to the file is updated and the file is deleted.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 13:15 Answer the three questions on axioms versus the other properties with one proposal: transparency, pickling and serialisation as `Equation`-valued laws of every carrier, so the matrix owns them and the property files go.
- [x] `proptest.yml`: `persist-credentials: false` on the checkout.
- [x] `assert_axioms` accepts a `NotImplemented` verdict, as `falsify` and the matrix do.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-03 13:15 Answer the `DIAGRAMS` thread: an empty parametrisation is an explicit pytest skip, and #659 enrols the diagram carriers.
- [x] Docstrings on the new named helpers the diff adds without one.
