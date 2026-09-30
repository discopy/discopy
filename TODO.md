# TODO

Prompted by a 🌙 Evening run, whose stored prompt is quoted verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

The head it picked is [#775](https://github.com/discopy/discopy/issues/775), `utils.dumps`
crashing on any box whose `data` carries a `complex`, ranked for tonight by the 🐦 Birdsong
run of 2026-09-29T04:2xZ.

- [x] Measure the extent of the crash on `main`, since the issue names `QuantumGate` and the
      fix is only as narrow as the defect: which boxes raise, and whether any of them is
      outside the quantum module
- [x] Choose between the issue's two repairs on that measurement and say on the pull request
      what would change our mind
- [x] Implement it, with the round trip stated as a law rather than as a list of gates
- [x] Tests: the gates of the issue, a box outside the quantum module, a nested `data`, and
      the collisions a JSON-level decoder can have with a box whose `data` is a dict
- [x] `CHANGELOG.md` under `[Unreleased]`
- [x] Report the bugs met on the way that are not this one, as `AGENTS.md` asks
- [ ] `pflake8 discopy`, `pylint discopy`, `coverage run -m pytest`, and the `proptest` matrix
- [ ] Delete this file, which clears the merge gate and hands the head to the reviewers
