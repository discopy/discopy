> Clean up the discopy changelog that accumulated after three months of vibe coding, condense it as much as possible, keeping only the difference wrt to the previous release (i.e. before we even had a changelog)

- [WIP] @session_01F6hn2cPwGcF56EfWivjiDo-2026-10-22 10:00 Rewrite `[Unreleased]` as the net difference with `1.2.2`: drop the fixes to features `1.2.2` never had, the CI work added then removed, the history of how each entry got there
- [ ] Add the breaking changes made before `CHANGELOG.md` existed: `closed` split from `biclosed`, `Category` removed, `ob_map`/`ob`/`ar`, Python ≥ 3.12
- [ ] One line per entry, references as link definitions at the bottom
