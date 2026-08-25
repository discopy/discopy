# Review round: cubic-dev-ai, 2026-08-24 16:12

Quoted verbatim from the two unresolved review threads on #586:

> `discopy/quantum/reservoir.py` — `Reservoir.__init__` allows negative
> `memory` or `inputs`, creating instances that are guaranteed to fail later;
> validate both as non-negative integers before checking the unitary shape.

> `test/quantum/ansatze.py` — the new Rydberg list-waveform path lacks a
> positive numerical test, so valid time-dependent behavior could regress
> undetected; add coverage for a correctly sized list waveform.

- [x] Validate `memory`/`inputs` as non-negative in `Reservoir.__init__`
- [x] Add a positive numerical test for the Rydberg list-waveform path
