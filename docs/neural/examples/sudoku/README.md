# Sudoku: three neural interpretations of one diagram

Three sudoku solvers built with [`discopy.neural`](../../../../discopy/neural/),
compared under one protocol: same data, same encoder and readout, same
optimizer, parameter counts matched within 10%. Only the *diagram*, the widths
on its wires, the cell and the **solver** differ, so a difference in the results
is a difference between the architectures.

## What a model is here

A model is a `MapNN` between an encoder and a readout, and a `MapNN` is three
things:

```python
model = MapNN(
    ob={MESSAGE: Dim(24), STATE: Dim(96), CLUE: Dim(24), ANSWER: Dim(0)},
    ar={"cell": Site(...), "unit": Relation(...)},
    solver=Iterate(rounds=64))
```

* **`ob`** — the width each atomic *role* carries. A role names what a wire
  carries (`message`, `state`, `clue`, `answer`), not how wide it is. Sending a
  role to `Dim(0)` erases its ports and the wires on them, which is how models
  A and C share one diagram.
* **`ar`** — one shared learnable module per generator name. All 81 cell sites
  are the *same* `Site` instance and all 27 unit sites the same `Relation`,
  which is what makes the model size independent of the grid size.
* **`solver`** — how the compiled diagram is executed.

Given a diagram, `MapNN.compile` turns the two into a global interaction
`T : S_D → S_D`, one round of synchronous message passing along the wires; the
solver says how many rounds to run and which are differentiated. The diagram
itself comes from the grid's combinatorics: `from_incidence(memberships(), …)`
for the bipartite cell/unit factor graph, `from_relation(peers_of(), …)` for the
pairwise clique. Training is then an ordinary PyTorch loop — see
[`train.py`](train.py).

A **trace** is a port wired to another port of the same box: private memory that
survives a round. Every cell here has at least two, a *state loop* and a *clue
loop*; model C adds an *answer loop*.

## The three models

| | diagram | wires | cell | solver |
|---|---|---|---|---|
| **A · GoI** | bipartite cell/unit factor graph | 405 | GRU `Site` + Deep-Sets `Relation` | `Iterate(rounds, inject=True)`, a loss on every round, one backward graph |
| **B · RRN** | pairwise peer clique (Palm et al. 2018) | 972 | LSTM `Site`, summed pair messages | same |
| **C · TRM** | A's diagram + traced answer loop | 486 | A's cell, resumable | `Recursion(rounds, cycles, steps)` (Jolicoeur-Martineau 2025) |

The wire counts are of the *messages*: model B's cell keeps the two states of
its `LSTMCell` as two named roles, `hidden` and `memory`, on one traced loop, so
its compiled map has 81 more traced wires than the 972 above — the same bytes in
the same places, named twice instead of sliced in two.

Best results on the held-out test split of the Palm et al. (2018) benchmark
(18,000 puzzles; 50k training puzzles, 8 epochs, mean over seeds 0–1):

| | cell | boards | boards at more test-time compute |
|---|---|---|---|
| A · GoI | 0.9842 | 0.8872 | **0.9182** at 144 rounds |
| B · RRN | 0.9456 | 0.7201 | 0.8293 at 288 rounds |
| C · TRM | 0.9750 | 0.8737 | — (1.2 GiB activations vs B's 13.7 GiB) |

Model A was trained at 64 rounds, lr 1e-3, batch 128, Adam, grad-norm clip 1.0.
Its bet is distance against wire count: a constraint is a single hyperedge over
nine variables rather than the 36 pairwise wires a clique needs to say the same
thing, so the map has 405 wires where B's has 972 — but a belief travels
cell → unit → cell, so one cell-to-cell hop costs two rounds here and one there.
The same architecture at 3e-3 and 20 rounds — what a one-epoch learning-rate
proxy had selected — reaches only 0.4107 boards and *degrades* past 32 test
rounds. The encoding was never the bottleneck; the step size was.

Model C's headline property is not accuracy but **memory**: 6 × 3 × 8 = 144
effective rounds per example in about 1.2 GiB of activations, against 13.7 GiB
for model B at comparable depth, because only the last cycle of each supervision
step is differentiated. That it can be cut into segments at all is a property of
the formalism, `T^(a+b) = T^b ∘ T^a`, and it needs the state at the cut to be
complete: the cell re-emits its clue instead of zeros (`resumable=True`), so the
run carries its own clues and message passing runs with `inject=False`.

Beyond the matched-budget comparison, two searched recipes are recorded in
[`config.py`](config.py) and run by `train.py`:

* `simple` — **0.9933** validation boards on the full Palm benchmark
  (`c-trm-v2` trial 5), at deliberately smaller widths than the matched ones;
* `extreme` — a 3×-width model on the much harder
  [sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
  benchmark (`trm-extreme-3x` trial 5): 0.4632 valid boards at trained depth,
  0.4801 at 32 supervision steps.

## Adaptive computation time, and the halt head as a verifier

`ACT` adds a halt head to model C: one logit per site, aggregated by a soft
minimum, trained to predict whether the current answer is already correct. At
training time [`ACTTrainer`](train.py) refills a batch slot with a fresh puzzle
the moment it halts; at inference `evaluate_act` stops each puzzle when its halt
logit turns positive.

The same head is a *verifier*. `evaluate_selected` runs independent stochastic
rollouts — Gaussian noise on the answer trace, one draw per supervision step —
and keeps the answer whose halt logit is largest, the only confidence signal
available at test time. On the `trm-extreme-act-8k` trial-2 checkpoint this took
the board rate from **0.674 to 0.894** at 256 supervision steps, with the halt
head performing close to an oracle verifier. [`evaluate.py --noise`](evaluate.py)
reproduces the whole depth-by-noise grid.

## Layout

    config.py     grid constants, paths, widths, budgets, recorded recipes
    dataset.py    the two benchmarks, the sudoku symmetry group
    model.py      roles, signatures, combinatorics, encoder/readout, the models
    train.py      the two supervision schemes, the registry, the recipes, ACT
    evaluate.py   fixed / adaptive / best-of-k / noise-sweep protocols
    optuna_act.py the ACT search on sudoku-extreme that selected the above

Everything else is `discopy.neural`'s and generic in the task. Datasets are
downloaded, verified and cached on first use under `../../sudoku_data/`; every
training run is checkpointed under `../../artifacts/`, and `train.py` re-loads a
finished run instead of re-training it.

## Running

    python train.py goi --seed 0          # a recorded baseline
    python train.py goi --seed 0 --quick  # a few-minute miniature
    python train.py simple                # the 0.9933-boards recipe
    python train.py extreme               # the sudoku-extreme recipe

    python evaluate.py ../../artifacts/best-goi-seed0.pt --model goi \
        --sweep 64 96 144
    python evaluate.py ../../artifacts/optuna-trm-extreme-act-8k-trial2.pt \
        --model act --extreme --noise

    python dataset.py                     # fetch and verify Palm et al. (2018)
    python dataset.py --extreme --check   # build and verify sudoku-extreme

    python optuna_act.py --gpus 2 --timeout 259200 \
        --seed-from <earlier.db> --seed-study trm-extreme-act-8k

The last is the search itself, three days of it across two GPUs sharing one
study. `--seed-from` copies the completed trials of an earlier study —
their hyperparameters, their values and their whole intermediate curves —
so the median pruner has something to compare against from the first trial
and the sampler starts from a posterior; the best imported configuration is
re-measured first. Every trial draws a fresh random seed, recorded in its
user attributes, so the search ranks configurations rather than seeds.

Run them from this directory. One GPU suffices; the maps train through
`CMap.forward` and speed up several-fold under `MapNN.compile_rounds`, which the
recorded recipes enable by default.

The figures under `../../figures/` were produced from the artifacts
`evaluate.py --noise` writes; the plotting script itself is not carried over.
