CLRS: neural algorithmic reasoning as three interpretations of one formalism

This document lives at docs/neural/CLRS/README.md and is the brief for the project. It is written to be executed in three parts, in order, by an implementer who has read docs/neural/README.md, ARCHITECTURE.md and the sudoku example. Each part is self-contained, has explicit done-criteria, and produces artefacts the next part consumes.

Goal

Port the sudoku methodology to a subset of the CLRS-30 algorithmic reasoning benchmark (Veličković et al. 2022): shared cells on a diagram drawn from an instance's combinatorics, compiled to a global interaction, run by a solver, supervised on the benchmark's per-step hints. Then use the machinery to answer a question the CLRS literature has not answered cleanly: which execution and differentiation policies — not which architecture — determine out-of- distribution algorithmic generalization, measured under one controlled protocol.

We are not chasing state of the art. The contribution is the instrument: one formalism spanning wiring, symmetry, solver policy and memory, with every promise measured (check_equivariant, Interaction.residual) rather than assumed.

Philosophy

Three commitments, inherited from the sudoku study:

A model is a diagram, an interpretation and a solver — nothing else varies uninspected. Same data, same encoder and readout, same optimizer, parameter counts matched within 10%; one axis differs per comparison, so a difference in results is a difference between named choices.
The task's artefacts live in the example, the machinery in the library. discopy/neural is not modified. The CLRS-specific code — combinatorics, signatures, encoders, readouts, losses, the directed-edge cell — lives in docs/neural/examples/clrs/, mirroring the sudoku layout (config.py, dataset.py, model.py, train.py, evaluate.py).
Claims are numbers. Equivariance is a measured residual; convergence is a measured residual; memory is a measured GiB; halting is calibrated against the ground-truth step count the benchmark provides.
The encoding

A CLRS sample over n nodes becomes a closed map via from_incidence:

Node boxes, name "node": a Site with Signature((Orbit(MSG, d, Sym.PERM), Orbit(STATE, traced=True))) — one message port per incident edge (plus one for the readout relation), a traced recurrent state.
Edge boxes, name "edge": one 2-member relation per graph edge. For undirected weighted graphs a stock Site suffices: Signature((Orbit(MSG, 2, Sym.PERM), Orbit(ESTATE, traced=True), Orbit(WEIGHT, traced=True))) with {ESTATE: Mode.STATE, WEIGHT: Mode.CARRY} — a Site emits a carried role unchanged, and Interaction.write(state, ("edge", WEIGHT), values) writes one value per edge, so per-edge inputs need no library change. Relation cannot play this part: it admits a single orbit and no traced roles.
Directed edges need a custom cell in model.py (~30 lines, subclassing cells.Cell): stock Site pools its message orbit symmetrically, so direction cannot live in a stock module. The cell reads (m_src, m_tgt, estate, weight) as distinguishable ports (Sym.NONE on the message orbit — an honest signature: check_equivariant is then not owed), updates the edge state, echoes the weight, answers each endpoint separately. from_incidence assigns edge member slots in node- index order, deterministically; the carry holds (w, orientation) where the orientation bit says whether the lower-indexed endpoint is the source.
Graph-level features and outputs: one "readout" relation containing every node, under its own name — the documented from_incidence pattern. Side effect: every node has degree ≥ 1, so isolated nodes in Erdős–Rényi samples never reach from_incidence's max().
Hints ↔ checkpoints. Iterate(rounds=R, deep=True, inject=False) with resumable/carried inputs; the caller decodes each checkpoint against the hint at the corresponding algorithm step and masks per-sample trajectory lengths. Open-loop only in Parts 1–2: no teacher forcing of hints back into the state (the post-Hint-ReLIC literature says closed-loop feeding often hurts; and open-loop needs no plumbing).
Decoders (caller-side): scalars/masks by linear heads on model.read(D, s, ("node", STATE)) or ("edge", ESTATE); pointers by a bilinear score over node-state pairs, softmaxed per row; graph-level features off the readout relation's state. One shared decoder per feature type, reused across algorithms.
Batching: intern one diagram per (algorithm, n, edge-set-hash) behind lru_cache is wrong — CLRS graphs differ per sample. Instead intern per (algorithm, n, m) histogram is still too fine. Practical rule: batch same-n samples (CLRS train is fixed small sizes), pad with Batch(..., pad=True) only across the OOD sweep. Degree heterogeneity is mild at CLRS scale; measure calls-per-round in Part 1 and report it.

Changes required inside discopy/neural: none. If anything small turns out to be genuinely forced, it must be ≤ ~10 lines, behaviour-preserving on sudoku (the golden gate in test/neural/ must stay green), and recorded in NOTES.md style: what, why, and why it is not a number.

The eight algorithms

Chosen for alignment with a synchronous message passer, with two deliberate probes of the boundary:

algorithm	why it is in
minimum	non-graph sanity check; the readout relation alone solves it
bfs	the canonical parallel wavefront; must saturate
bellman_ford	parallel relaxation; literally a fixed-point iteration
dag_shortest_paths	parallel DP with masking; mid-difficulty anchor
dijkstra	boundary probe: sequential frontier, yet MPNNs score high
mst_prim	boundary probe: sequential growth, mask_one hints
floyd_warshall	edge-level DP; the stateful-edge-box showcase (H1)
matrix_chain_order	interval DP; second edge/pair-state showcase (H1)

Excluded on principle: strings, geometry, sorting, DFS — sequential control flow misaligned with a synchronous equivariant cell; every published GNN processor collapses there OOD and no solver policy rescues a misaligned model class. Do not add them.

Research question and hypotheses

RQ: holding the diagram, the cells and the parameter budget fixed, how do (i) structural state placement, (ii) solver execution policy, (iii) differentiation policy and (iv) measured symmetry each covary with OOD generalization on aligned algorithmic tasks?

H1 (alignment). A traced edge state recovers a large fraction of the gap between plain MPNN and triplet-style edge reasoning on floyd_warshall and matrix_chain_order, because pairwise memory is the capability triplets approximate.
H2 (fixed points). On algorithms that are fixed-point iterations (bellman_ford, floyd_warshall), FixedPoint matches Iterate at trained depth and degrades less when test-time depth is swept, and its residual tracks the ground-truth algorithm's convergence step.
H3 (halting). An ACT halt head supervised with the benchmark's true trajectory length calibrates: halt step ≈ true step count, and F1-at-halt ≈ F1-at-oracle-depth at lower average compute.
H4 (laws). Across tasks and seeds, the check_equivariant residual of the trained cells correlates with the in-distribution → OOD F1 drop.

H2–H4 are where the library can stand out: no incumbent holds these numbers, because no incumbent separates execution policy from differentiation policy from architecture, or measures its symmetry.

Fair setup, and what to beat

Canonical CLRS-30 protocol: train on the benchmark's samples at n ≤ 16, validate at n = 16, test OOD at n = 64; metric is micro-F1 on outputs (hints supervise, hints are not scored); single-task training, one model per algorithm; match the compared paper's data budget exactly and say so.

Two anchors, both from Ibarz et al. 2022 ("A Generalist Neural Algorithmic Learner"), single-task columns:

Floor (must beat): vanilla MPNN with hints. Especially on floyd_warshall, where MPNN is weak — this is H1's target.
Ceiling (match within a few points, not beat): Triplet-GMPNN. Approximate OOD F1 values, to be transcribed exactly from the paper's table before any comparison is claimed — the figures below are from memory and marked as such: bfs ≈ 99.7, bellman_ford ≈ 97, dijkstra ≈ 96, dag_shortest_paths ≈ 88, mst_prim ≈ 86, minimum ≈ 98, matrix_chain_order ≈ 91, floyd_warshall ≈ 48.

Honest expectation: match the ceiling on bfs/minimum/bellman_ford, land between floor and ceiling elsewhere, and make the real claims in Part 3, where there is no incumbent.

The controlled comparison

Sudoku discipline, verbatim:

One table per axis; everything else frozen. Axes: state placement (node-only vs node+edge state — Dim(0) on ESTATE/WEIGHT erases the edge machinery from the same diagram, the sudoku A/C trick), solver (Iterate(deep) / FixedPoint / Recursion / ACT), depth at test time (sweep rounds ×1, ×1.5, ×3 of trained depth).
Parameter counts within 10% across compared rows; widths adjusted, never architecture.
Solver rows compared twice: at matched effective depth and at matched backward memory — these are different questions and conflating them is the standard sin.
≥ 3 seeds, mean ± std; GPU runs are compared over seeds only; any before/after refactor check runs on CPU, single thread, per NOTES.md.
Every number lands in a JSON artefact beside its provenance, sudoku-style.
Part 1 — the harness, and proof of life

Scope: minimum, bfs, bellman_ford. Everything hard here is plumbing; build it once, correctly.

Deliverables, mirroring examples/sudoku:

dataset.py: load CLRS-30 trajectories (the clrs package or its pre-generated files), expose (diagram_key, inputs, hints, lengths, outputs); download, verify, cache.
model.py: roles, the two signatures, build_diagram(algorithm, edges) via from_incidence with the readout relation; encoders for the feature types these three tasks need (scalar, mask, pointer; node/graph); the shared decoders; undirected Site edge boxes only (BFS/BF need no direction — for bellman_ford on directed CLRS graphs, symmetrize in Part 1 and note it; Part 2 fixes it properly).
train.py: Iterate(deep=True) hint+output loss with per-sample length masks; the registry/recipe pattern.
evaluate.py: OOD F1 at n = 64, per feature type and micro-averaged; test-time round sweep.
test_clrs_smoke.py: end-to-end on a miniature in seconds, in the spirit of test_sudoku_smoke.py.

Metrics: in-distribution F1 (must saturate on all three), OOD F1 vs the two anchors, wall-clock per epoch, batched-calls-per-round.

Done when: bfs OOD ≥ 0.99; bellman_ford OOD ≥ 0.90 with a plausible path to the ceiling; the smoke test runs in CI time; no change to discopy/neural was needed (or the exception is documented).

Part 2 — edge state, direction, and the full table

Scope: add dijkstra, dag_shortest_paths, mst_prim, floyd_warshall, matrix_chain_order. The scientific payload is H1.

The directed-edge cell in model.py (the ~30-line custom Cell); mask_one hint decoding for dijkstra/mst_prim (decode, don't feed back).
floyd_warshall and matrix_chain_order on the complete-graph diagram with stateful edge boxes; matrix_chain_order's intervals are node pairs — same shape.
The H1 ablation: each showcase task trained twice from the same diagram — ESTATE → Dim(0) (node-only) vs ESTATE → Dim(w) — parameters rebalanced to match. Report both against floor and ceiling.
The full 8 × {ours, floor, ceiling} table, with the exact published numbers transcribed.

Metrics: the table; the H1 delta; per-step hint F1 curves (where in the trajectory does the model diverge from the algorithm — diagnostic gold).

Done when: all eight tasks train under one protocol; node-only vs edge-state is a clean two-column result on the two showcase tasks; every row beats the floor.

Part 3 — solvers, residuals, halting, laws

Scope: the study that stands out; no new tasks. The deliverable is a results section, not a feature.

H2: on bellman_ford and floyd_warshall, train under Iterate and FixedPoint (both backward modes) at matched parameters; sweep test-time depth; plot Interaction.residual per step against the ground-truth algorithm's convergence step (computable from the hints). The claim to test: the learned map is contractive exactly where the algorithm is, and FixedPoint is depth-robust where Iterate(deep) oscillates.
Memory: Recursion vs one-graph Iterate at matched effective depth at n = 64: activation GiB vs OOD F1 — the sudoku 1.2-vs-13.7 plot, re-drawn on a benchmark where depth is semantically meaningful. Stretch: n = 128, where one backward graph stops fitting and Recursion does not.
H3: ACT with the halt head supervised by the true trajectory length (a signal sudoku never had). Report MAE(halt step, true steps), F1-at-halt vs F1-at-fixed-budget, mean compute saved.
H4: check_equivariant residual of every trained cell (float64), per task and seed, against the ID→OOD F1 drop; report the correlation with its uncertainty. Small-sample caveat stated, not hidden: 8 tasks × 3 seeds is suggestive, not conclusive.

Metrics: residual curves; GiB-vs-F1; halt calibration; the H4 scatter.

Done when: each of H2–H4 has a figure, a table and a one-paragraph verdict, including negative verdicts — a hypothesis cleanly refuted under a controlled protocol is a result.
