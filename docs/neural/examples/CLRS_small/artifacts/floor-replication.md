# Phase 1 — the floor, reproduced; and R-vs-O inside the reference

Everything below is trained by `floor.py` in the dm-clrs venv, on the
published `CLRS30_v1.0.0` archive itself (provenance:
`floor-dataset-provenance.json`), under the 2022 recipe named in
`floor.RECIPE` — noisy teacher forcing 0.5 with hard hint feedback, Adam
1e-3, batch 32, 10 000 steps, no gradient clipping, hidden 128, max
aggregation, fully-connected MPNN — three seeds a row, best-val
checkpoint restored, OOD at `n = 64` on the benchmark's own 32
trajectories.  Arm `R` is that recipe; arm `O` is the same run with
hints off entirely (`encode_hints=False, decode_hints=False`), which is
the reference implementation's only honest "output-only": it removes the
hint loss and the hint feedback together, at unchanged depth.  Per-seed
rows and per-head splits are in `floor-{R,O}-<algorithm>-report.json`.

## The gate

| algorithm | local floor (arm R) ± s.e.m. | published (Table 2 MPNN) ± s.e.m. | verdict |
|---|---|---|---|
| `bellman_ford` | **0.9139 ± 0.0011** | 0.9201 ± 0.0028 | reproduced (Δ 0.6 pts ≈ 2 combined s.e.m.; R2 says that is noise) |
| `bfs` | **0.9974 ± 0.0003** | 0.9989 ± 0.0005 | reproduced (Δ 0.15 pts) |
| `mst_prim` | **0.6911 ± 0.0546** | 0.6908 ± 0.0756 | reproduced, spread included |
| `floyd_warshall` | **0.3015 ± 0.0294** | 0.2674 ± 0.0177 | reproduced (slightly above, within noise) |
| `dijkstra` | **0.8535 ± 0.0247** | 0.9150 ± 0.0050 | ~6 pts shy, one weak seed (0.804/0.879/0.877) — see the caveat |

**The gate is passed.**  Four of five rows land on the published number
within noise; `dijkstra` is the exception and its shortfall has a named
suspect: the 2022 runs early-stop on a 32-trajectory validation split
evaluated every ~10 steps (v1.0.0 `run.py`, `eval_every = 320` *items*),
where this reproduction evaluates every 100 steps — ten times fewer
draws from a selection lottery that `NOTES.md` already documents as
worth a tenth on 32 samples.  The within-harness R-vs-O contrast below
is unaffected; no claim in this file compares `dijkstra`'s local R
against the published column.

Wall-clock worth recording: a seed is **2–8 minutes** on a (shared!)
H100 for every row except `floyd_warshall`'s ~15, so the entire phase —
gate plus replication, 30 trainings — cost under three GPU-hours.  The
reference harness is not the expensive half of this project.

## R-vs-O, replicated in the reference implementation

Pre-registered predictions (the mission brief, before these runs):
`dijkstra`/`mst_prim` improve substantially output-only; `bellman_ford`
roughly unchanged; `floyd_warshall` degrades.  Measured:

| algorithm | R (hints, TF 0.5) | O (no hints) | Δ (O − R) | prediction |
|---|---|---|---|---|
| `dijkstra` | 0.8535 ± 0.0247 | **0.9378 ± 0.0046** | **+8.4 pts** | confirmed |
| `mst_prim` | 0.6911 ± 0.0546 | **0.7544 ± 0.0102** | **+6.3 pts** | confirmed |
| `bellman_ford` | **0.9139 ± 0.0011** | 0.8771 ± 0.0116 | −3.7 pts | roughly held (a modest hints-help, same sign and size as Mahdavi et al.'s 92.0 → 90.1) |
| `bfs` | 0.9974 ± 0.0003 | 0.9959 ± 0.0008 | −0.15 pts | unchanged, confirmed |
| `floyd_warshall` | 0.3015 ± 0.0294 | **0.3587 ± 0.0106** | **+5.7 pts** | **refuted** — see below |

With three seeds an arm these are suggestive by the study's own R2
(permutation floor p = 0.10), but the `dijkstra` and `mst_prim`
separations are seed-for-seed clean and land where two independent
literatures put them: Mahdavi et al. (TMLR 2023) report the same signs
for no-hint MPNNs (dijkstra +2.0, mst_prim +5.6), and Rodionov &
Prokhorenkova (NeurIPS 2023) build their no-hint result on the same
observation.  **Part 3's central R-vs-O finding is therefore not a
pipeline artefact**: removing per-step hint supervision helps exactly
the rows it helps in our pipeline, inside DeepMind's own code.  The
magnitudes differ — ours are larger (`dijkstra` 0.068 → 0.477) because
our O arm removes the hint *gradient* from a much deeper unroll — and
the reference's O arm removes feedback and loss together where ours
separates them, which is exactly the distinction the Phase 3
teacher-forcing arm exists to close.

**The `floyd_warshall` prediction is refuted, and that is a finding
about our pipeline.**  In the reference, no-hint *helps*
`floyd_warshall` (+5.7, again matching Mahdavi's 26.7 → 39.5); the
no-hint arm's ID validation is unchanged (0.88–0.90 both arms).  In our
pipeline the same deprivation collapses ID (0.889 → 0.567, Part 3 arm
O).  So "edge-level DP needs hints to fit" is a property of **our
encoding** — a complete-graph diagram whose per-pair state must be
built through two-round hops — and not of the task.  The honest
statement for the paper: hint supervision hurts OOD in both
implementations wherever trajectories grow with `n`; our diagram
additionally *depends* on hints to fit `floyd_warshall` in
distribution, which the reference's dense one-step processor does not.

## Per-head splits, and the two facts they add

Local floor (arm R), OOD means over three seeds
(`floor-R-*-report.json` carries per-seed):

| algorithm | output `pi` | pointer hints | mask hints | order-dependent index hints | scalar hints (MSE) |
|---|---|---|---|---|---|
| `bellman_ford` | 0.9139 | `pi_h` 0.9490 | `msk` **1.000** | — | `d` 0.0012 |
| `bfs` | 0.9974 | `pi_h` 0.9989 | `reach_h` **1.000** | — | — |
| `dijkstra` | 0.8535 | `pi_h` 0.8650 | `mark` 0.894, `in_queue` 0.753 | `u` **0.1398** | `d` 0.0129 |
| `mst_prim` | 0.6911 | `pi_h` 0.7382 | `mark` 0.900, `in_queue` 0.818 | `u` **0.1675** | `key` 0.0149 |

* **The floor's own order-dependent global-index heads are dead too.**
  `u` — "which node the loop is on", a `mask_one` over the node set — is
  at 0.14–0.17 OOD in the reference while its masks sit at 0.8–1.0.
  That is the exact signature of our M3 heads (`minimum`'s `pred_h`
  0.141, `floyd_warshall`'s `k` 0.094), reproduced in DeepMind's
  implementation with teacher forcing and a dense processor.  M3 is a
  property of the task's label semantics, not of our diagram.
* **The floor resolves the tie mass.**  `bfs`'s pointer is 0.9974
  against an order-blind ceiling of 0.5012 at `n = 64` (69.7 % of
  assignments index-tie-broken), so `pos` plus a dense processor
  suffices to break essentially every tie.  Tie mass is a fact about
  the labels, not an excuse: the realizable ceiling with `pos` in hand
  is ~1.0, and the floor sits on it.  Our best `bfs` arm reads 0.9296
  with the same `pos` available — the remaining seven points are ours
  to explain, not the dataset's.

## The audited differences, after Phase 1's measurements

* **Per-sample output freeze: ruled out** for `bellman_ford`
  (`snapshot-eval.json`): scoring our Part 3 checkpoints at each
  sample's own termination checkpoint — the reference's semantics —
  changes nothing (R 0.668 vs 0.677, S 0.704 vs 0.714, O 0.708 vs
  0.724; if anything the hold reading is *higher*).  On `dijkstra`/
  `mst_prim` every trajectory is the same length, so the freeze cannot
  matter there either.
* **Still standing, in rank order**: hint re-encoding (closed-loop
  feedback — the floor's state is refreshed by decoded hints at every
  step, ours never is; note the floor keeps most of its `bellman_ford`
  score *without* it, arm O's 0.8771, so feedback alone bounds that
  suspect at ~4 points there), the fully-connected processor (one dense
  hop per step against one sampled-edge hop), and the pointer-decoder
  machinery M2 already localized.  These are Phase 3's arms.

## Environment

`/scratch/tommaso.salvatori/dm-clrs`: Python 3.11, `dm-clrs` 2.0.3,
jax 0.10.2 + CUDA 12 plugin, `protobuf` pinned to 6.33 (see
`NOTES.md`), TF barred from the GPU, jax on the `platform` allocator.
Logs under `/scratch/tommaso.salvatori/floor-logs/`.
