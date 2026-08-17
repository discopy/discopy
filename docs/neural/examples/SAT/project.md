Project: One Wiring, Many Policies — a controlled study of learned message passing on random SAT/MaxSAT

Overall goal. Not to build a competitive SAT solver (that framing loses). The claim we are building toward: for learned message passing on SAT, the structural choices — hyperedge factor graph vs. pairwise clique, execution/differentiation policy, test-time compute — determine solve rates and their scaling, and this can be measured cleanly because all variants are interpretations of diagrams with shared generators under one protocol. The deliverable is a set of scaling curves and symmetry measurements, plus one practically useful result: exact-verifier best-of-k rollouts as free test-time compute for MaxSAT.

Overall philosophy. discopy.neural separates what is usually entangled: the diagram (structure), the interpretation (widths + shared modules), the solver (execution + differentiation policy), and the task (encoder/readout/loss, ours). A CNF formula is a bipartite factor graph, so each instance is a diagram; the dataset is a distribution over diagrams built from three generator names (lit, clause, flip). Because everything else is held fixed, a difference in results is attributable to a named axis — that attribution is the science. Honesty rules: report decode rates against trivial baselines (WalkSAT, greedy), never claim competitiveness with CDCL, treat non-convergence near the phase transition as a measurable phenomenon, not a bug.

Part 1 — Encoding, pipeline, and the factor-graph baseline

Goal. A working end-to-end pipeline: random k-SAT instances → diagrams → trained MapNN → per-variable assignments, with decode rate curves. Everything later builds on this.

Encoding in discopy.neural. NeuroSAT-style literal splitting, because the wiring carries no per-edge signs and state is addressed per site, not per port. Each variable becomes two literal nodes; polarity lives in the graph:

python
MSG, STATE, CSTATE = Ty("msg"), Ty("state"), Ty("cstate")
lit    = Signature((Orbit(MSG, 3, Sym.PERM), Orbit(STATE, traced=True)))
clause = Signature((Orbit(MSG, 3, Sym.PERM), Orbit(CSTATE, traced=True)))
flip   = Signature((Orbit(MSG, 2, Sym.PERM),))

Build with from_incidence: incidence lists, per literal node, the indices of its clauses plus its flip relation; relation={"clause": clause_sig, "flip": flip_sig}, relation_name per relation (this per-relation naming is explicitly supported — see the "readout" example in the docstring). Fill lit with a Site (GRU, mean pool), clause with a Relation (Deep Sets, sum pool) — or a Site with a traced CSTATE loop for a stateful clause, the NeuroSAT-faithful variant; implement both, flag-switched. Intern diagrams per instance behind an lru_cache keyed on the incidence tuple; batch heterogeneous instances with Batch(..., pad=True) and bucket.

Task layer (ours, outside the library). No clues: the formula is the diagram, so the initial state is a learned constant embedding (plus small noise) written to the state traces; no inject. Readout: model.read(D, state, ("lit", STATE)) → linear head → per-literal logit; variable probability from the positive literal (consistency with the negative literal can be a small auxiliary loss). Loss: unsupervised smooth-SAT — clause score 1 − ∏(1 − p_lit), loss −Σ log(score + ε). No assignment supervision (SAT solutions are non-unique; supervising one is ill-posed). Decode by thresholding at 0.5; an instance is solved iff the rounded assignment satisfies every clause — an exact, free check.

Data. Random 3-SAT: uniform n ∈ [10, 100], α swept over {3.0, 3.5, 4.0, 4.26}; ~100k train instances (generation is trivial), held-out test sets at each (n, α), plus larger n ∈ {150, 200, 300} kept exclusively for Part 2's generalization study. Keep only satisfiable instances for training (filter with any classical solver — instant at these sizes).

Solver. Iterate(rounds=32, inject=False) with deep=True, loss summed over per-round checkpoints (the sudoku model-A recipe).

Experiments and metrics. (i) Train, sweep rounds at test time (32→256): decode rate (fraction of instances solved) and clause-satisfaction rate vs. rounds, per (n, α). (ii) Baselines on identical test sets: random assignment, greedy, WalkSAT at matched wall-clock. (iii) Sanity: loss curves, and check_equivariant on both cells confirming lax equivariance (residual ≈ float noise). Success criterion: decode rate well above WalkSAT-at-matched-flops is not required; decode rate materially above greedy at α ≤ 4.0, n ≤ 100, with monotone improvement in test rounds, is. That establishes the instrument works.

Part 2 — The controlled comparison: structure × policy × hardness

Goal. The core scientific content: attribute performance differences to named axes, on a fixed protocol inherited from Part 1.

Research question. Which matters more for learned SAT message passing — the diagram (hyperedge factor graph vs. pairwise literal clique), or the solver (one differentiated run vs. segmented recursion)? And how does each degrade with instance size and with proximity to the satisfiability threshold?

Hypothesis (falsifiable, mirrors the sudoku findings). (H1) The factor graph beats the clique at matched parameters and degrades more gracefully under test-time round extension, because a hyperedge states a k-ary constraint in k wires instead of k(k−1) pairwise ones, at the price of two rounds per variable-to-variable hop. (H2) Recursion matches Iterate accuracy at a fraction of the activation memory, enabling deeper effective unrolls that pay off at larger n. (H3) All variants show a sharp decode-rate cliff as α → 4.26 accompanied by rising FixedPoint residuals — learned message passing inherits BP's non-convergence geometry near condensation.

Encoding of the second diagram. The clique: from_relation over the literal–literal graph where two literals are related iff they co-occur in a clause (plus the flip pairs), literal cell = Site with LSTM and per_leg=True — the model-B analogue. Caution: co-occurrence degrees vary more than factor-graph degrees; measure the number of distinct-degree groups per batch and the resulting throughput (this is itself a reportable cost axis).

Experiments. A 2×2×(+1): {factor graph, clique} × {Iterate, Recursion(rounds, cycles, steps)} at parameter counts matched within 10%, plus FixedPoint diagnostics on trained weights. For each: (i) decode rate vs. test rounds/supervision steps, per (n, α); (ii) size generalization — train n ≤ 100, test n ∈ {150, 200, 300}; (iii) residual ‖T(s) − s‖∞ vs. α; (iv) activation memory and wall-clock per effective round; (v) equivariance residuals across the grid. Two seeds minimum, means reported; this is where the GPU hours go, and every run is embarrassingly parallel.

Metrics. Decode rate; clause-sat rate; rounds-to-solve distribution; memory (GiB) at matched effective depth; degradation slope in n; residual-vs-α curves. The paper figure is decode-rate-vs-test-compute with one line per (diagram, solver) — the SAT analogue of the sudoku A/B/C table, with the phase-transition panel as the physics hook.

Part 3 — Test-time compute with an exact verifier, and MaxSAT

Goal. The practically interesting result, cheap to obtain given Part 2's best checkpoint: exact verification makes best-of-k rollouts free lunch, and the same machinery extends to weighted MaxSAT.

Philosophy. The sudoku study needed a learned halt head as verifier (0.674 → 0.894 via best-of-k). For SAT the verifier is exact and costless: count satisfied clauses. So the entire test-time-compute apparatus — ACT halting, noisy rollouts, selection — comes with its guarantee built in. This is the one place the domain is kinder than sudoku.

Encoding. Take Part 2's winning (diagram, solver); add the answer trace via the Dim(0) trick — one diagram, erased answer role for the plain model, Dim(y) for the Recursion/ACT model with resumable=True cells (the model-C move, verbatim). For MaxSAT: weighted smooth-SAT loss −Σ w_c log(score_c + ε) on weighted random instances (unit weights + a weighted tail); everything else unchanged.

Experiments. (i) Best-of-k: k ∈ {1, 4, 16, 64} independent rollouts (Gaussian noise on the answer trace, per the sudoku noise protocol), exact-verifier selection; decode rate (SAT) and mean satisfied weight fraction (MaxSAT) vs. total test compute. (ii) ACT: halt head trained to predict current-assignment correctness (exact labels — again free); adaptive vs. fixed compute at matched average rounds; compute-savings histogram. (iii) MaxSAT honesty check: satisfied-weight fraction vs. greedy, vs. WalkSAT/SATLike at matched wall-clock — expect to lose to SATLike; report it. (iv) Optional stretch if time remains: model's assignment as warm start for a local-search MaxSAT solver, metric = time-to-target vs. random init. Metrics: decode rate / satisfied-weight vs. total compute (the scaling curve is the claim); ACT compute-accuracy Pareto; warm-start speedup ratio.

The final claim, assembled from all three parts: on random SAT/MaxSAT, structure and policy — not scale — govern how learned message passing converts test-time compute into solutions; the factor-graph + recursion combination scales furthest, all variants hit the same wall at the satisfiability threshold where their fixed-point residuals diverge, and exact verification makes rollout-based test-time scaling essentially free.