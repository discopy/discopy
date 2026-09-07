# TODO

> torch is off the egress allowlist
>
> let's fix that now!

- [x] A `SessionStart` hook for Claude Code on the web that installs the full development environment, falling back to PyPI's `torch` wheel when `download.pytorch.org` is not reachable
- [ ] Register it in `.claude/settings.json` — the session's permission classifier refused to write that file twice; a human adds the 14-line registration, or allows the write
- [x] Validate: the hook runs, `pflake8` runs, one test runs, the full suite runs with torch (910 passed, 2 skipped)
- [x] `CHANGELOG.md` entry
