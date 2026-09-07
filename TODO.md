# TODO

> torch is off the egress allowlist
>
> let's fix that now!

- [ ] A `SessionStart` hook for Claude Code on the web that installs the full development environment, falling back to PyPI's `torch` wheel when `download.pytorch.org` is not reachable
- [ ] Register it in `.claude/settings.json`
- [ ] Validate: the hook runs, `pflake8` runs, one test runs, the full suite runs with torch
- [ ] `CHANGELOG.md` entry
