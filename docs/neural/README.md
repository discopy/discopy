# Neural networks as combinatorial maps

Three sudoku solvers built with [`discopy.neural`](../../discopy/neural/),
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

The maps themselves are built in two stages, syntax then semantics. A
`Signature` (`sudoku/signature.py`) says what ports a box has — how many,
grouped into orbits, which of them are traced, and which are one orbit
under a permutation. From it, `discopy.neural.skeleton` builds an
abstract, torch-free *skeleton* out of the grid's combinatorics
(`sudoku/skeleton.py`): a closed map whose atomic types name the *role* of
each port rather than its width. An `Interpretation`
(`discopy.neural.functor`) then sends each role to the `Dim` it carries and
each box name to the torch module computing it, and `interpret` applies it.
Since `Dim(0)` is the monoidal unit, an interpretation can erase a role's
ports altogether — which is how models A and C share one skeleton: A sends
the answer role to `Dim(0)`, C to `Dim(48)`.

The signature is also the *only* place a port offset is written down. The
abstract type of a box, the loop wiring the skeleton lays down and the flat
slices a module reads and writes are all derived from it, so they cannot
drift apart; and it is what a symmetry is declared on, which
`check_equivariant` then measures numerically. A learned cell is only
*laxly* structured — permutation-equivariance holds up to the reordering of
a floating-point sum, Frobenius fusion does not hold at all, and
`cells.fusion_residual` reports how far off it is rather than claiming
otherwise.

## The three models

| | map | wires | update cell | supervision |
|---|---|---|---|---|
| **A · GoI** (`sudoku/train_a_goi.py`) | bipartite cell/unit factor graph | 405 | GRU site + Deep-Sets relation | every round, one backward graph |
| **B · RRN** (`sudoku/train_b_rrn.py`) | pairwise peer clique (Palm et al. 2018) | 972 | LSTM site, summed pair messages | every round, one backward graph |
| **C · TRM** (`sudoku/train_c_trm.py`) | A's map + traced answer loop | 486 | A's site, resumable | per detached segment (Jolicoeur-Martineau 2025) |

The wire counts are of the *messages*: model B's cell keeps the two states
of its `LSTMCell` as two named roles, `hidden` and `memory`, on one traced
loop, so its map has 81 more traced wires than the 972 above — the same
bytes in the same places, named twice instead of sliced in two.

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

The *method* is not here at all: the wiring, the cells, the interpretation
and the message-passing schedules live in
[`discopy.neural`](../../discopy/neural/), where they are generic in the
task and in the source category. What is left in this folder is a *study*:
`core/` holds the harness that trains and scores a model, and `sudoku/`
brings only what is irreducibly sudoku — the grid combinatorics, the roles
its wires carry, an encoder, a decoder, the two benchmarks and the recorded
configurations. There is no cell class and no solver class here; a model is
a choice of skeleton, widths and schedule. A future task adds a sibling
package with its own combinatorics and data, and configures the same
engine.

    core/                 the benchmark kit (see core/__init__.py)
      study.py            the torch-free dataclasses: Widths, Budget, Split
      train.py            the harness: deep supervision, evaluation, batching
      heads.py            the fill-in-the-blanks encoder and decoder
      solvers.py          the three solver shapes, parts order written once
      registry.py         checkpoints, cached training, lr grid, per TaskSpec
      act.py              the ACT evaluations bound to the decode rule
      recipes.py          optimizer, schedule, EMA, segmented loop
    sudoku/               the sudoku task (see sudoku/__init__.py)
      config.py           grid constants, paths, budgets, matched widths
      signature.py        the roles a port can play + the box signatures
      skeleton.py         rows/columns/blocks/peers -> the two skeletons
      heads.py            historical import path over core.heads
      data.py             the Palm et al. (2018) benchmark + symmetry group
      sudoku_extreme.py   the sudoku-extreme benchmark, three variants
      models.py           models A, B, C on the core.solvers templates
      act.py              historical import path over core.act
      train.py            the TaskSpec + historical entry points
      train_a_goi.py      model A: recorded best configuration + protocol
      train_b_rrn.py      model B: likewise
      train_c_trm.py      model C: likewise
      best/               the optuna-winner recipes, on core.recipes
    golden/               the frozen pre-refactor fingerprints + recorder
    migration.py          loading a checkpoint trained before the refactor
    NOTES.md              what was left alone, and why
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
several-fold under `CMap.compile` (see `Engine.compile_cells`), which the
`best/` recipes enable by default.
