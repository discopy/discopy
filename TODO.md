# TODO.md

> let's open a separate PR with the linter failing on unused variables, imports, etc. either fixing the corresponding bugs or adding exceptions explicitly,
>
> also another idea could be to get some way of checking the diff in pylint output so that we can prevent the situation from getting worse, simplest is to fail under the current score then take each category of warnings and work through them or deactivate them if not relevant

— toumix on [discopy#767](https://github.com/discopy/discopy/pull/767#discussion_r4041370727),
2026-09-17 20:56 UTC.

- [ ] `.pylintrc`: `fail-on` the unused family (`unused-import`, `unused-variable`,
      `unused-argument`, `unused-wildcard-import`, `unused-private-member`,
      `possibly-unused-variable`), `fail-under` at the score the `lint` job reports on this head,
      and the deprecated `suggestion-mode` option pylint reports as `E0015` dropped
- [ ] the 42 unused arguments on `main`: fixed where the parameter should be read, an explicit
      exception with its reason where the interface requires it
- [ ] the 16 unused imports and 7 unused variables on `main`: same
- [ ] `AGENTS.md` and `CONTRIBUTING.md` name `uv run pylint discopy` beside `pflake8`, so the
      job's red is seen before a push
- [ ] changelog; the remaining warning categories counted for the follow-up; `pflake8` and the
      suite green
