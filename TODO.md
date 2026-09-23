> also another idea could be to get some way of checking the diff in pylint output so that we can prevent the situation from getting worse, simplest is to fail under the current score then take each category of warnings and work through them or deactivate them if not relevant

— toumix, 2026-09-17 20:56, on the review thread of #767. The gate is #768, landed on 2026-09-23 ahead of #767, which its threshold turns red for the first category listed on #770.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-23 13:15 `no-self-argument` and `no-member` on the laws: a pylint plugin, `.github/scripts/pylint_axioms.py`, reading a function decorated with `@axiom` as the classmethod it is, loaded from `.pylintrc` and tested under `.github/tests`
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-23 13:15 `no-member` on the factories, set by `NamedGeneric`'s subscript or by a module after its class: `generated-members`
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-23 13:15 `py-version` 3.10 for code that requires 3.12, and `import-outside-toplevel` on `hypothesis.strategies`, imported where a strategy is built so that the package never imports it: the two settings
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-23 13:15 `fail-under` raised to the score of `main` under the new configuration, measured in the job's environment; CONTRIBUTING.md and the changelog
