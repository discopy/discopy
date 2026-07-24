# Neural networks as combinatorial maps

Three sudoku solvers built with [`discopy.neural`](../../discopy/neural.py),
compared under one protocol: same data, same embedding and readout, same
optimizer, parameter counts matched within 10%. Only the *wiring* of the map,
the width on its wires, the update cell and — for one model — the evaluation
strategy differ, so a difference in the results is a difference between the
architectures.

## What a model is here

All three models are the same kind of object: a **closed combinatorial
map**. That means two things and nothing else.

1. A finite family of *boxes*. A box is a `Network(name, dom, cod, module)`:
   a PyTorch module together with a list of typed ports, where a port
   carries a `Dim(w)` — a wire of width `w`. Boxes are **shared**: the same
   module instance appears at many sites of the map, which is what makes the
   model size independent of the grid size.

2. A *fixpoint-free involution* on the set of all ports — an edge relation
   pairing each port with exactly one other, never itself.
   `CMap.from_wiring` takes the boxes and that pairing, given as
   `(box_index, port_position)` endpoints.

Running the map for one **round** is then completely determined: every box
reads its in-ports, its module runs, it writes its out-ports, and the
involution σ permutes those emissions into the next round's inputs. This is
the execution formula of the geometry of interaction, `m ↦ σ ⊕ᵢ fᵢ(m)`.
Repeated rounds compose, `F^(a+b) = F^b ∘ F^a`, which is the law model C
exploits to run its recursion in resumable segments.

A **trace** is a port wired to another port of the same box: private memory
that survives a round. Every cell here has at least two, a *state loop* and
a *clue loop*; model C adds an *answer loop*.

The maps themselves are built in two stages, syntax then semantics: an
abstract, torch-free *skeleton* (`core/skeletons.py`, fed the grid's
combinatorics by `sudoku/skeleton.py`) fixes who talks to whom, with
atomic types naming the *role* of each port rather than its width; a
`discopy.neural.Functor` (`core/functors.py`) then sends each role to the
`Dim` it carries and each abstract box to the `Network` computing it,
applied by `core.cmaps.interpret`. Since `Dim(0)` is the monoidal unit, a
functor can erase a role's ports altogether — which is how models A and C
share one skeleton: A's functor sends the answer role to `Dim(0)`, C's to
`Dim(48)`.

## The three models

| | map | wires | update cell | supervision |
|---|---|---|---|---|
| **A · GoI** (`sudoku/train_a_goi.py`) | bipartite cell/unit factor graph | 405 | GRU cell + Deep-Sets unit | every round, one backward graph |
| **B · RRN** (`sudoku/train_b_rrn.py`) | pairwise peer clique (Palm et al. 2018) | 972 | LSTM cell, summed pair messages | every round, one backward graph |
| **C · TRM** (`sudoku/train_c_trm.py`) | A's map + traced answer loop | 486 | A's cell, resumable | per detached segment (Jolicoeur-Martineau 2025) |

Best results on the held-out test split of the Palm et al. (2018) benchmark
(18,000 puzzles; 50k training puzzles, 8 epochs, mean over seeds 0–1 —
budget details and caveats in each script's docstring):

| | cell | boards | boards at more test-time compute |
|---|---|---|---|
| A · GoI | 0.9842 | 0.8872 | **0.9182** at 144 rounds |
| B · RRN | 0.9456 | 0.7201 | 0.8293 at 288 rounds |
| C · TRM | 0.9750 | 0.8737 | — (1.2 GiB activations vs B's 13.7 GiB) |

Beyond the matched-budget comparison, `sudoku/best/` records the strongest
recipes found by the optuna searches in [`../optuna/`](../optuna/):
`simple_sudoku_trm.py` reaches **0.9933** validation boards on the full
benchmark, and `extreme_sudoku_trm.py` trains a 3×-width model on the much
harder [sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
benchmark (0.4632 valid boards at trained depth, 0.4801 at 32 supervision
steps).

## Layout

The folder separates the *method* from the *instance*: `core/` is the
general library — the solver family and everything needed to build, train
and evaluate it — and `sudoku/` brings only what is irreducibly sudoku:
the grid combinatorics, the two benchmarks, and the recorded
configurations of the study. A future task adds a sibling package with
its own combinatorics and data, and instantiates the same solvers.

    core/                 the general library (see core/__init__.py)
      study.py            the torch-free dataclasses: Widths, Budget, Split
      cmaps.py            map readers, interpret, resumable routing
      skeletons.py        port roles + the factor-graph and clique shapes
      functors.py         the functors filling a skeleton in, Layout
      solvers.py          the cells + the three solver architectures
      act.py              adaptive computation time for the recursion solver
      train.py            the harness: deep supervision, evaluation, batching
      recipes.py          optimizer, schedule, EMA, segmented loop
    sudoku/               the sudoku task (see sudoku/__init__.py)
      config.py           grid constants, paths, budgets, matched widths
      skeleton.py         rows/columns/blocks/peers -> the two skeletons
      data.py             the Palm et al. (2018) benchmark + symmetry group
      sudoku_extreme.py   the sudoku-extreme benchmark, three variants
      models.py           models A, B, C = the family bound to the skeletons
      act.py              model C with the halt head
      train.py            the study protocol: checkpoints, registry, lr grid
      train_a_goi.py      model A: recorded best configuration + protocol
      train_b_rrn.py      model B: likewise
      train_c_trm.py      model C: likewise
      best/               the optuna-winner recipes, on core.recipes
    artifacts/            checkpoints and cached results   (gitignored)
    sudoku_data/          the two benchmarks, fetched on first use (gitignored)
    figures/              figures written by the notebooks  (gitignored)

Every dataset is downloaded, verified and cached on first `load()`; every
training run is checkpointed under `artifacts/`, and the `train_*.py`
scripts re-load a finished run instead of re-training it. The notebooks in
[`../notebooks/`](../notebooks/) — `neural-functors.ipynb` for the formalism,
`neural-cells-lecture.ipynb` for a close-up of model C — import this
folder's packages.

## Running

    python sudoku/train_a_goi.py --seed 0          # full recorded budget
    python sudoku/train_a_goi.py --seed 0 --quick  # few-minute miniature
    python sudoku/best/simple_sudoku_trm.py        # the 0.9933-boards recipe

One GPU suffices; the maps train through `CMap.forward` and speed up
several-fold under `CMap.compile` (see `Solver.compile_cells`), which the
`best/` recipes enable by default.
