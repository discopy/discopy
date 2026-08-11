# -*- coding: utf-8 -*-

"""
The noise study's latent statistics, on a model small enough for CI.

``evaluate.latent_stats`` is what the study reads *before* it injects
anything: it reports the scale of the two carried states, which is what
lets ``sigma`` be read as a relative noise level.  It wants a trained
checkpoint and a benchmark to be interesting, neither of which is in the
repository -- but it runs on any model of the family, so it runs on a tiny
one, and what is testable without the checkpoint is exactly what used to
break: the widths it reads are read off the interpretation that is
executing, not repeated at the call site.
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

import evaluate as evaluations                          # noqa: E402
import model as zoo                                     # noqa: E402
from config import Widths                               # noqa: E402

#: Small enough that a supervision step is instant, wide enough that the
#: answer and the latent have different widths -- which is the whole point,
#: since the bug was reading one of them off the wrong object.
TINY = Widths(dim=4, state_dim=8, hidden=8, y_dim=4)

#: The scalars :func:`latent_stats` reports for each carried state.
SCALARS = ("global_mean", "per_feature_std_mean", "per_feature_std_min",
           "per_feature_std_max", "rms", "mean_norm")


@pytest.fixture(scope="module")
def model():
    """ The halting recursion of the study, at test widths. """
    torch.manual_seed(0)
    return zoo.act(TINY, rounds=1, cycles=2, steps=2,
                   halt_detach=True, halt_head="softmin").eval()


@pytest.fixture(scope="module")
def clues():
    """ Two puzzles of the committed golden fixture. """
    golden = SUDOKU.parents[1] / "golden" / "act.npz"
    return torch.as_tensor(np.load(golden)["clues"][:2], dtype=torch.long)


def test_the_widths_come_from_the_interpretation(model):
    """
    The invariant the statistics rest on: the widths are read off the
    compiled interaction that is executing, so they cannot drift from a
    :class:`~config.Widths` repeated at a call site.
    """
    found = model.interaction
    for key, width in ((("cell", zoo.ANSWER), TINY.y_dim),
                       (("cell", zoo.STATE), TINY.state_dim)):
        assert found.widths[found.heads[key][0]] == width


def test_latent_stats_of_a_tiny_model(model, clues):
    """
    One supervision step of statistics: an answer block and a latent
    block, each with finite scalars, and each at its own width.
    """
    stats = evaluations.latent_stats(model, clues, steps=1)
    assert sorted(stats) == ["y", "z"]
    for block in stats.values():
        assert all(np.isfinite(block[key]) for key in SCALARS)
        assert block["rms"] > 0.0


def test_the_noiseless_sweep_is_deterministic(model, clues):
    """
    ``sigma = 0`` is the deterministic run the study scores everything else
    against, so it must not depend on the generator it is handed:
    :func:`~evaluate.run_segment` skips the draw entirely.
    """
    target = clues.clone()
    caps = (1, 2)

    def board(generator):
        book = evaluations.fresh_book(model, clues)
        out = evaluations.run_segment(
            model, clues, target, 0.0, caps, generator, 0.0, book, max(caps))
        return out[max(caps)]["fixed_correct"].float().mean().item()

    assert board(None) == board(torch.Generator().manual_seed(1))


def test_a_sweep_records_how_it_was_produced(model, clues, tmp_path):
    """
    *How* a sweep was produced belongs beside *what* was asked of it:
    eager and compiled agree only up to the rounding freedom
    ``CMap.compile`` documents, so the torch version and the device are
    recorded with the numbers.
    """
    from dataset import Split
    split = Split("smoke", clues.numpy(), clues.numpy())
    payload = evaluations.noise_sweep(
        model, split, caps=(1, 2), sigmas=(0.0, 0.5), rollouts=2,
        survivors=1, batch_size=2, log=lambda *a, **k: None)
    assert set(payload["results"]) == {1, 2}
    assert payload["protocol"]["rollouts"] == 2
    for row in payload["results"].values():
        for box in row.values():
            assert 0.0 <= box["fixed"]["best_of_k"] <= 1.0
            assert box["fixed"]["pass_at_k"] >= box["fixed"]["single"] - 1e-9
    text = evaluations.table(payload["results"], (1, 2), (0.0, 0.5),
                             "fixed", "best_of_k")
    assert text.splitlines()[0].split() == ["steps", "0", "0.5"]

    monkey = evaluations.ARTIFACTS
    try:
        evaluations.ARTIFACTS = tmp_path
        json_path, npz_path = evaluations.write_sweep(payload, "smoke")
    finally:
        evaluations.ARTIFACTS = monkey
    import json
    summary = json.loads(json_path.read_text())
    assert summary["environment"]["torch"] == torch.__version__
    assert "raw" not in summary and npz_path.exists()
