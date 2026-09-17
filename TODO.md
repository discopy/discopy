# Review round of 2026-09-17: CodeRabbit on `TermBase.generate`

> **Expose the term-generation procedures as methods.**
>
> `TermBase.generate` declares eight local procedures, including closures inside `applications` and `abstractions`. STYLE.md requires each subprocedure to be exposed as a testable, reusable method. Move this logic to a named generator object or equivalent method-based abstraction, and keep `generate` as a small entry point. The nesting guidance supports this refactor, but does not specifically require a generator object.

— coderabbitai, 2026-09-17 17:29 UTC, on `discopy/biclosed.py` line 673.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-17 17:35 `Sampler`, a dataclass holding the choices, the types, the letters and the counter, with one method per procedure — `choose`, `constant`, `variable`, `bound`, `spine`, `leaves`, `splits`, `application`, `applications`, `abstraction`, `term` — `TermBase.generate` reduced to building one and asking it for a term, the methods unit-tested
