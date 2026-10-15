# TODO

Prompt (toumix, 2026-10-15, interactive):

> Keep the discopy-next bootstrap going, spawn subagents to fix the issues you have filed and give me an executive summary of places where there are high-level design decisions to be made then prepare the first batch of PRs to go from discopy-next back upstream

- [ ] cut slice 3 of rel-int/discopy-next#55 from `main`: `discopy.quantum.reservoir` and the `Rydberg` ansatz, stacked on slice 2 (`claude/slice-2-small-fixes`) since the `Reservoir` doctest needs #517's `Controlled` fix
- [ ] CI green on the real jobs
