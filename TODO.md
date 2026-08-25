# Review round: cubic-dev-ai, 2026-08-24

Feedback quoted verbatim from the two unresolved review comments on
`discopy/quantum/reservoir.py`:

> P2: When `memory` or `inputs` is negative, `qubit ** (...)` treats the
> count as the unit, so construction succeeds and `run`/`step` later fail
> on incompatible circuits. Validate both counts as non-negative integers
> before building the expected wire type. (line 68)

> P2: When `unitary` contains a mixed box, this constructor accepts it if
> its endpoints are qubits, so `step` is no longer guaranteed to be
> induced by a unitary or CPTP map. Reject mixed circuits with
> `unitary.is_mixed`. (line 69)

- [ ] `Reservoir.__init__` rejects negative `memory`/`inputs`
- [ ] `Reservoir.__init__` rejects a mixed `unitary`
- [ ] tests for both
