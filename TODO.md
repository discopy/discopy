# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.
>
> — scheduled prompt of the `evening` routine, fired 2026-10-12 00:06:31 UTC

🌙 took [#744](https://github.com/discopy/discopy/pull/744) as tonight's head. Its own branch
`refactor/serde` is not this session's to push to ([desire#46](https://github.com/daydream6728/desire/issues/46),
un-reacted since 2026-10-10), so the repair is authored here, on the branch this session *was*
assigned, and offered as one click into `refactor/serde`.

- [ ] delete the three conflict markers committed into `CHANGELOG.md` at `f30d7b28`
- [ ] merge `main` into the branch, resolving `CHANGELOG.md` by keeping both sides
- [ ] assert no marker survives anywhere in the tree
- [ ] validate: `pflake8`, `pylint` against the `fail-under` the head inherits, `pytest`
- [ ] check the tree the push produces is the tree that was validated
