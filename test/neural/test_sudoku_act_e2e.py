# -*- coding: utf-8 -*-

"""
End to end on the real benchmark: train model C with ACT, then read it
with the noise study's own machinery.

``test_sudoku_smoke.py`` runs the same production code paths on eight
puzzles at toy widths -- enough to pin shapes, detach boundaries and slot
refill, not enough to tell a solver from a random tensor.  This file is the
other half: the Palm et al. (2018) benchmark, the matched widths of the
study, and a training budget long enough that the model actually learns to
solve sudoku.  Only then do the interesting assertions become available --
that the carried answer is unit-scale because the refresh normalises it,
and that injecting noise into it *costs* board accuracy, which on an
untrained model is unfalsifiable.

The evaluation is the study's, not a reimplementation:
:func:`~evaluate.latent_stats` and :func:`~evaluate.run_segment` are
imported from ``docs/neural/examples/sudoku``, so this test guards the
example as well as the library.  What it does not do is the beam schedule
-- rollouts, survivor selection, pass@k -- which is the slow part and adds
no coverage of the library.

Cost and gating.  Roughly three and a half minutes on one H100, almost all
of it the training loop, so this is not a test to run on every commit: it
is marked ``neural_e2e`` and it skips unless a GPU is present and the
benchmark is already cached, since a test must never reach for the network.
Run it when ``discopy.neural`` changes::

    pytest -m neural_e2e test/neural/test_sudoku_act_e2e.py

The thresholds below are deliberately slack -- roughly a third of what the
budget actually reaches -- because this is an integration test, not a
score.  The measured run at 6000 iterations gives cell 0.895, board 0.438
and a noise curve of 0.428 / 0.404 / 0.156 at sigma 0 / 0.1 / 0.5.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SUDOKU = (Path(__file__).resolve().parents[2]
          / "docs" / "neural" / "examples" / "sudoku")
if str(SUDOKU) not in sys.path:
    sys.path.insert(0, str(SUDOKU))

import dataset                                          # noqa: E402
import evaluate as evaluations                          # noqa: E402
import model as zoo                                     # noqa: E402
import train as training                                # noqa: E402
from config import DATA_DIR, WIDTHS                     # noqa: E402

#: The benchmark, as ``dataset`` caches it once fetched.
CACHED = all((DATA_DIR / f"{split}.npz").exists()
             for split in ("train", "valid"))

pytestmark = [
    pytest.mark.neural_e2e,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="the training budget needs a GPU"),
    pytest.mark.skipif(not CACHED,
                       reason=f"benchmark not cached under {DATA_DIR}; "
                              "run dataset.fetch() once"),
]

#: The budget: enough optimizer steps to learn, few enough to be a test.
SEED, N_TRAIN, N_VALID = 0, 5000, 1000
BATCH_SIZE, ITERATIONS = 256, 5000

#: The depths, as ``config.QUICK`` sets them for model C.
ROUNDS, CYCLES, N_SUP = 4, 2, 4

#: What the budget must clear, at roughly a third of what it reaches.
MIN_CELL, MIN_BOARD = 0.70, 0.15

#: The noise levels of the sweep: none, and enough to hurt.
SIGMAS = (0.0, 0.5)


@pytest.fixture(scope="module")
def trained():
    """
    Model C trained with the real ACT loop: the halt head, the paper's
    loss and the slot refill, on the benchmark's training split.

    Returns ``(model, valid split, first stats, last stats)``.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    splits = dataset.load(verify=False)
    valid = splits["valid"].subsample(N_VALID)

    model = zoo.act(WIDTHS["trm"], rounds=ROUNDS, cycles=CYCLES,
                    steps=N_SUP, halt_detach=True,
                    halt_head="softmin").to(device)
    train = splits["train"].subsample(N_TRAIN)
    clues, targets = training.to_device(train, device)
    stream = training.ExampleStream(
        clues, targets, np.random.default_rng(SEED))
    optimizer = training.adamw(model, lr=1e-3, weight_decay=1e-2)
    scheduler = training.cosine_schedule(
        optimizer, warmup=ITERATIONS // 10, total=ITERATIONS)
    trainer = training.ACTTrainer(model, stream, BATCH_SIZE)

    first = trainer.run(optimizer, scheduler, None, iterations=20)
    last = trainer.run(optimizer, scheduler, None,
                       iterations=ITERATIONS - 20)
    return model, valid, first, last


def clues_and_solution(split, device, count):
    """
    A device-resident batch, the solution left **one**-indexed: the study
    scores ``Decoder.decode(logits, clues)``, which writes the clues back
    over ``argmax + 1``, so it compares against the raw solution.  Only the
    cross-entropy of the training loop wants it zero-indexed.
    """
    return (torch.as_tensor(split.puzzles[:count], dtype=torch.long,
                            device=device),
            torch.as_tensor(split.solutions[:count], dtype=torch.long,
                            device=device))


def test_act_training_learns_to_solve(trained):
    """
    The whole recursion trained through the ACT loop: the loss falls, no
    parameter goes non-finite, and the model solves boards it has never
    seen -- which is what makes the noise assertions below meaningful.
    """
    model, valid, first, last = trained
    for key in ("loss", "ce", "q", "depth"):
        assert np.isfinite(first[key]) and np.isfinite(last[key]), key
    assert last["loss"] < first["loss"]
    assert 1.0 <= last["depth"] <= N_SUP
    assert all(torch.isfinite(value).all()
               for value in model.state_dict().values())

    scores = evaluations.evaluate(model, valid, batch_size=500)
    assert scores["cell"] > MIN_CELL
    assert scores["board"] > MIN_BOARD


def test_latent_stats_of_a_trained_model(trained):
    """
    The statistics the study reads before it injects anything.  On a
    trained model they are no longer merely finite: the refresh normalises
    the carried answer at the end of every cycle, so ``y`` is unit-scale
    per cell, which is what lets ``sigma`` be read as a relative noise
    level.
    """
    model, valid, _, _ = trained
    clues, _ = clues_and_solution(valid, torch.device("cuda"), 500)
    stats = evaluations.latent_stats(model, clues, steps=N_SUP)

    assert sorted(stats) == ["y", "z"]
    for block in stats.values():
        assert np.isfinite(block["rms"]) and block["rms"] > 0.0
        assert np.isfinite(block["per_feature_std_mean"])
    assert 0.5 < stats["y"]["rms"] < 3.0


def board_at(model, clues, target, sigma, generator):
    """ Board accuracy after :data:`N_SUP` noisy supervision steps. """
    book = evaluations.fresh_book(model, clues)
    out = evaluations.run_segment(
        model, clues, target, sigma, (N_SUP, ), generator, 0.0, book, N_SUP)
    return out[N_SUP]["fixed_correct"].float().mean().item()


def test_latent_noise_costs_board_accuracy(trained):
    """
    The study's finding, in the small: noise on the answer trace is not
    free.  Injected once per supervision step at a level comparable to the
    state's own scale, it costs boards the deterministic run solves.
    """
    model, valid, _, _ = trained
    device = torch.device("cuda")
    clues, target = clues_and_solution(valid, device, 500)
    boards = {
        sigma: board_at(model, clues, target, sigma,
                        torch.Generator(device=device).manual_seed(SEED))
        for sigma in SIGMAS}

    assert all(0.0 <= board <= 1.0 for board in boards.values())
    assert boards[0.0] > MIN_BOARD
    assert boards[0.0] > boards[0.5]


def test_the_noiseless_sweep_is_deterministic(trained):
    """
    ``sigma = 0`` is the deterministic run the study scores everything else
    against, so it must not depend on the generator it is handed.
    """
    model, valid, _, _ = trained
    device = torch.device("cuda")
    clues, target = clues_and_solution(valid, device, 500)
    once = board_at(model, clues, target, 0.0, None)
    twice = board_at(model, clues, target, 0.0,
                     torch.Generator(device=device).manual_seed(SEED + 1))
    assert once == twice


def test_the_perturbation_keeps_the_loop_consistent(trained):
    """
    The two ports of a cell's answer loop intentionally carry the same
    ``y``, so one noise sample per cell is written to both ends: the noise
    perturbs the state, it does not break the loop.
    """
    model, valid, _, _ = trained
    device = torch.device("cuda")
    clues, _ = clues_and_solution(valid, device, 8)
    found = model.interaction
    state = model.initial(clues)
    generator = torch.Generator(device=device).manual_seed(SEED)
    noisy = evaluations.perturb_answer(model, state, 0.5, generator)
    every = found.read(noisy, model.answer, every=True)
    assert every.shape[1] == 2 * model.n_cells
    assert torch.equal(every[:, 0::2], every[:, 1::2])
    assert not torch.equal(found.read(noisy, model.answer),
                           found.read(state, model.answer))
