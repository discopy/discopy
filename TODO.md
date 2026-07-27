# TODO

Prompt (Alexis, verbatim):

> let's move the skill away from discopy into my personal repo

- [x] Remove `.claude/skills/bob/` — `.claude/` is now gone from the repo entirely
- [x] Repoint RULES.md rule 4 at the copy in `toumix/toumix.github.io`
- [x] `uv run pflake8 discopy` clean. `uv run pytest` still aborts with the 8 collection errors
      `main` already has without the extras installed — unchanged by this diff, and what
      [#479](https://github.com/discopy/discopy/pull/479) fixes
