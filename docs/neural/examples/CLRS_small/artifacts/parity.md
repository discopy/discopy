# Phase 0 — recipe parity: ours against the floor's, row by row

The floor every table is measured against is one number per algorithm, and
Phase 0's question is what *recipe* produced it and where ours differs.
Every claim below carries a file and a line on both sides.  "Ours" is the
Part 3 protocol (`config.PART3`, the `p3-max-ptredge-*` campaign);
"reference" is `clrs` 2.0.3 in the rebuilt venv at
`/scratch/tommaso.salvatori/dm-clrs` (see the environment note at the end),
read beside the v1.0.0 `run.py` fetched from the GitHub tag and the two
papers' own text.

## The floor's provenance, settled

The floor is **Veličković et al. 2022, Table 2, MPNN column** — for
`bellman_ford`, **92.01 ± 0.28** over 3 seeds.  Ibarz et al. 2022 Table 2
reprints those columns *"taken directly from Veličković et al. [2022]"*
and contains **no MPNN-with-improvements column anywhere**: the only model
trained with the Section-3 improvements is Triplet-GMPNN (bellman_ford
97.39 ± 0.19, 10 seeds).  So the floor's recipe is the 2022 one and
includes **none** of the Ibarz-era stabilizers — no randomized position
scalar, no soft hint propagation, no static-hint elimination, no Xavier
scalar-encoder init, no gradient clipping, no gating, no triplets, no
mixed training sizes, no on-line data augmentation.  What it *does*
include, from the 2022 paper's own text: noisy teacher forcing at 0.5,
hint re-encoding at every step, max aggregation, a fully-connected
processor, the fixed 1000-trajectory `n = 16` dataset, Adam at 1e-3 for
10 000 steps, batch 32, early stopping on validation.

One label caveat: Ibarz et al. state their error bars are *"standard
error of the mean across seeds (3 seeds for previous SOTA experiments)"*,
which is what `config.ANCHORS` records as `sem`; the 2022 paper's own
caption reads as a std over 3 seeds.  The digits are the same either way;
which statistic the `±` names for the reprinted columns is a
transcription ambiguity worth one line, not a finding.

## The table

| row | ours | reference floor | verdict |
|---|---|---|---|
| **a. `pos`** | sampler's `arange(n)/n`, train and eval, untouched (`Budget.pos="sampler"`, `config.py:301`; `dataset.py:160`; verified on the cached splits) | identical: `A_pos = np.arange(A.shape[0])`, `'pos': A_pos/n` — `clrs/_src/algorithms/graphs.py:58,64` (`bellman_ford`), `:183,189` (`bfs`).  Randomized pos is Ibarz §3.2.2 and 2.0.3's `run.py:46` default, **not** the floor's | **SAME** |
| **b. teacher forcing / hint feedback** | open-loop: hints enter the loss only (`model.py:1700–1778`); nothing re-encodes them into the state (`Model.encode`, `model.py:1480–1505`, encodes input probes alone) | hints are **encoded as inputs at every step**, train and eval (`nets.py:124–166`).  Train: per-sample Bernoulli, ground truth w.p. `hint_teacher_forcing` — the paper's 0.5 — else the model's own **hard**-postprocessed previous prediction (`baselines.py:154`; `decoders.py:159–176`).  Eval: its own predictions fed back (`nets.py:133–136`).  The initial condition `hints[0]` is fed as an input at the first step in both regimes | **DIFF — the largest structural difference.**  The closed-loop arm `PART3.md` names is still unrun (no artefact exists) |
| **c. loss forms** | per type: scalar MSE, mask masked-BCE, mask_one/categorical softmax-CE, pointer row-CE (`model.py:980–1131`); finished samples masked by `step+1 < lengths` (`model.py:1687`) | same forms (`losses.py:86–115` outputs, `:165–203` hints); same mask, `lengths > i+1` (`losses.py:206–210`) | **forms SAME**; averaging & placement DIFF, next row |
| **c′. loss placement** | output supervised at **every checkpoint from a sample's termination onward**, mean over steps (`model.py:1752–1763,1773`); hints: per-step per-probe mean over alive rows, ÷ steps (`model.py:1764–1774`); `hint_weight = 1` (`config.py:102`) | output decoded **once**, each sample's prediction **frozen at its own termination step** (`nets.py:175–181`) and the loss taken there (`baselines.py:425–430`); hints pooled over time×alive per probe then summed (`losses.py:144–162`), weight 1 implicitly (`baselines.py:432–437`) | **DIFF, deliberate** (NOTES.md, "two differences… both deliberate") — but see row f: the freeze also *exempts the floor from ever holding an answer* |
| **d. optimizer** | AdamW `wd=0` ≡ Adam, lr 1e-3 (`train.py:257`, `config.py:287–288`); 300×32 = 9600 steps (`config.py:373`); clip **1.0** (`config.py:98`, `train.py:112`); torch default init; best-val kept every 10 epochs = 320 steps (`train.py:266–291`, `config.py:291`) | Adam lr 1e-3, batch 32, 10 000 steps, *"early stopping on the validation performance"* (paper §; v1.0.0 `run.py:337–349` keeps and restores best-val, evaluated every 320 **items** = 10 steps); **no clipping** (clipping is Ibarz improvement 5); haiku LeCun-style init | **SAME** in optimizer/lr/batch/steps/selection; **DIFF-minor**: ours clips at 1.0 (an extra stabilizer on our side), init family differs, ref evaluates val 32× more often |
| **d′. capacity** | dim 16 / state 96 / hidden 192 / edge 48 / graph 96 (`config.py:392–397`); LayerNorm in every cell (`discopy/neural/cells.py:254`, `model.py:435`); **418 514** params (`bellman_ford`), 380–418 k depending on head | hidden 128 everywhere; LN on (`use_ln` default True in v1.0.0 `run.py:81` and 2.0.3 `run.py:97`); no gating; **414 468** params (`bellman_ford`), 413 827 (`bfs`) — counted by instantiating `BaselineModel` with the mpnn processor in the venv | **SAME within 1 %** (R4 satisfied) |
| **e. evaluation** | `DECODERS[…].score` verbatim transcriptions; split pooled then scored once (`model.py:1804–1853`); hint step `k` scored against `hints[k+1]` (`model.py:252`) | `postprocess(hard=True)` → `evaluation.evaluate`; pooled split (`collect_and_eval`); `evaluate_hints` `idx=i+1` (`evaluation.py:111–116,137–138`; `decoders.py:159–176`) | **SAME — verified on a shared prediction file**: pointer & mask_one **bitwise**, scalar equal in float64 (1.8e-15), mask differs only by the reference's own float32 accumulation (2.7e-8).  Artefact: `parity-eval-xcheck.json` |
| **f. rounds per hint step** | **2 rounds per checkpoint** (`HOPS`, `model.py:107`; checkpoint after round `2(k+1)`, `alignment`, `model.py:206–249`) = exactly **one neighbour exchange along sampled edges** + one global exchange through the readout relation; run length `max(lengths)` steps | **1 processor step per hint step** (`nets.py:255`, `nb_mp_steps = T−1`), and each step is a **fully-connected** message pass: `MPNN(PGN)` sets `adj_mat = jnp.ones_like(adj_mat)` (`processors.py:515–521`); the sampled graph enters as an edge *feature* only.  Run length `max(lengths)−1` steps | **DIFF on three counts, not one.**  (i) information radius per step: whole graph vs one sampled-graph hop; (ii) the per-sample output freeze (row c′) means the floor is never iterated past a sample's own termination — it is structurally exempt from M1; (iii) hint re-injection (row b) refreshes its state from outside the recurrence every step.  The "2 rounds = 1 exchange" bookkeeping itself is hop-for-hop **matched** to the reference's per-step neighbourhood *only on the complete-graph diagrams*; on sparse diagrams ours is strictly slower per step |
| **g. data budget** | 1000 train @ `n=16` seed 1 / 32 val @ 16 seed 2 / 32 test @ 64 seed 3 (`config.py:40–44`), drawn by `clrs.build_sampler` at those settings (`dataset.py:544–546`); Part 3 trains **fixed** `n=16` on `bellman_ford`, `dijkstra`, `mst_prim`, `floyd_warshall` and mixed only on `dag_shortest_paths` (`artifacts/regime.json`); wide-128 split seed 30 reported beside the canonical 32 | the same `CLRS30` spec verbatim (`samplers.py:46–61`); the paper: *"1,000 trajectories for training… inputs of 16 nodes"* — single size, **not** mixed; ER `p` fixed at 0.5 (`samplers.py:558`, `p=(0.5,)`) | **SAME** (for the Part-3 rows; the `MIXED` regime was tried and rejected per row, which restores this parity).  One residual risk: our cache is drawn with 2.0.3's samplers, the published splits with v1.0.0's — determinism across versions is unverified; the cheap check is a diff of our cached test split against the tfds `CLRS30` download, queued for the start of Phase 1 |

## What Phase 3 should take from this

Ranked by expected leverage on the `bellman_ford` gap (0.7302 arm S / 0.7323
arm O against 0.9201), the audited differences are:

1. **Hint re-encoding + teacher forcing (row b)** — the one wholesale
   regime difference, already earmarked in `PART3.md` as the last
   structural difference, and never run.  Note the tension to report
   either way: our own Part 3 shows the per-step hint *gradient* is what
   breaks depth extrapolation (O beats R on `dijkstra`/`mst_prim`), while
   the floor benefits from hints as *inputs* at `n = 64`.  Feeding hints
   forward and supervising them are different channels; the arm separates
   them.
2. **Per-sample output freeze + dense processor (rows c′/f)** — the floor
   answers each sample at its own termination step through a
   fully-connected pass; ours must hold answers to batch-max depth through
   a sparse diagram.  T-C's rounds-per-step arm addresses only a fraction
   of this; the freeze is an *evaluation-semantics* difference that could
   be measured on our side by scoring each sample at its own
   `HOPS·length` checkpoint — cheap, evaluation-only, and it isolates
   "holding the answer" from "computing it".
3. **Everything else is parity or minor** — pos, losses' forms, optimizer,
   capacity, eval semantics, data.  The clip-1.0 and init differences are
   noise-level suspects and are on the conservative side (ours has the
   extra stabilizer).

## Environment note, and the one thing the brief got wrong

The brief said the dm-clrs venv was already built.  It was not — no
interpreter on this machine could import `clrs` (all 13 conda envs, all 4
venvs, home and scratch searched).  It is now at
`/scratch/tommaso.salvatori/dm-clrs` (Python 3.11, `clrs` 2.0.3, jax
0.10.2 **CPU**; a CUDA jaxlib is a Phase-1 install).  Version caveat
carried by every reference citation above: 2.0.3 is Ibarz-era code whose
*defaults* are their recipe (`run.py:33–107`: `train_lengths 4–16`,
`random_pos True`, `hint_teacher_forcing 0.0`, `triplet_gmpnn`); the 2022
floor recipe is reachable from it by flags
(`--processor_type mpnn --train_lengths 16 --random_pos=False
--hint_teacher_forcing 0.5 --hint_repred_mode hard
--encoder_init default --grad_clip_max_norm 0.0 --learning_rate 0.001`)
plus the fixed dataset, which is Phase 1's job to pin.

## The cross-check, reproducibly

`parity-eval-xcheck.json` holds both scorings of one shared prediction
file (32 trajectories, `n = 64`, every probe type, seeded).  To re-run:
the three scripts are in the session scratchpad under `xcheck/`
(`make_shared.py`, `score_ours.py` in the study's venv,
`score_reference.py` in the dm-clrs venv, the latter with and without
`JAX_ENABLE_X64=1`).
