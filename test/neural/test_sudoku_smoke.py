# -*- coding: utf-8 -*-

"""
End to end, small: the real sudoku example on a tiny model.

``test_equivalence.py`` freezes the *numbers* of the four recorded models
and ``test_general.py`` holds the library's capabilities.  What neither
does is run a training loop from end to end, so a change that breaks the
way the pieces are wired together -- the segmented recursion, the refresh,
the detach boundaries, the halt head, the slot refill, the three evaluation
protocols -- would only show up in a script nobody runs in CI.

This file closes that gap.  Everything is the production path:
``docs/neural/examples/sudoku``'s own ``model``, ``train`` and ``evaluate``.
Only the *size* is a test's: the same 9x9 factor graph and peer clique at
widths small enough that a whole training loop is a second.  The batch is
the first eight puzzles of the committed golden fixture, so nothing is
downloaded.

The assertions are deterministic on purpose -- shapes, finiteness, which
parameters receive a gradient and which do not, a checkpoint round trip --
rather than an accuracy that would make CI a coin flip.  The one
optimisation claim, that the loss goes down on a fixed batch, is a smoke
assertion beside them, not the criterion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

NEURAL = Path(__file__).resolve().parents[2] / "docs" / "neural"
SUDOKU = NEURAL / "examples" / "sudoku"
if str(SUDOKU) not in sys.path:
    sys.path.insert(0, str(SUDOKU))

import evaluate as evaluations                          # noqa: E402
import model as zoo                                     # noqa: E402
import train as training                                # noqa: E402
from config import Budget, Widths                       # noqa: E402
from dataset import Split                               # noqa: E402
from model import Decoder                               # noqa: E402

#: Small enough that a whole training loop is a second, large enough that
#: every module of every model is exercised at its real shape.
TINY = Widths(dim=4, state_dim=8, hidden=8, y_dim=4)

#: The depths: one round per cycle and two cycles per supervision step, so
#: that a recursion step has exactly one undifferentiated cycle and one
#: differentiated one -- the detach boundary the tests below pin down.
ROUNDS, CYCLES, N_SUP = 1, 2, 2

#: The batch: the first eight puzzles of the committed golden fixture.
BATCH = 8

BUDGET = Budget(name="smoke", n_train=BATCH, n_valid=BATCH, n_test=BATCH,
                epochs=1, batch_size=4, rounds=ROUNDS, cycles=CYCLES,
                steps=N_SUP)


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def puzzles():
    """ The golden batch as a :class:`Split`, one-indexed solutions. """
    arrays = np.load(NEURAL / "golden" / "act.npz")
    return Split("smoke", arrays["clues"][:BATCH],
                 arrays["target"][:BATCH] + 1)


def tensors(split):
    """ The split as ``(clues, zero-indexed targets)``. """
    return (torch.as_tensor(split.puzzles, dtype=torch.long),
            torch.as_tensor(split.solutions, dtype=torch.long) - 1)


def build(name: str, seed: int = 0, **kwargs):
    """ One tiny model of the zoo, freshly seeded. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return zoo.build(name, BUDGET, widths=TINY, **kwargs)


def finite(model) -> bool:
    return all(torch.isfinite(value).all()
               for value in model.state_dict().values())


# --- every model of the zoo runs on the same batch -------------------------

@pytest.mark.parametrize("name", ["goi", "rrn", "trm", "act"])
def test_forward_of_every_model(name, puzzles):
    """
    Fixed-compute execution of each configuration: the shared interface
    ``model(clues)`` gives one logit per cell per class, at every
    supervised checkpoint under ``deep``.
    """
    model = build(name)
    clues, _ = tensors(puzzles)
    with torch.no_grad():
        logits = model(clues)
        every = model(clues, deep=True)
    assert logits.shape == (BATCH, 81, 9)
    assert torch.isfinite(logits).all()
    assert len(every) == (model.n_sup if model.segmented else model.rounds)
    assert torch.equal(every[-1], logits)
    assert evaluations.evaluate(model, puzzles, batch_size=4)["cell"] >= 0.0
    assert model.n_cells == 81


def test_the_two_models_share_one_diagram():
    """
    Models A and C read the *same* diagram object; only the width of the
    answer role differs, and ``Dim(0)`` erases it.
    """
    assert build("goi").diagram is build("trm").diagram
    assert ("cell", zoo.ANSWER) not in build("goi").interaction.heads
    assert ("cell", zoo.ANSWER) in build("trm").interaction.heads


# --- the two supervision schemes, through the real epoch harness -----------

@pytest.mark.parametrize("name,opt_steps", [("goi", 2), ("trm", 4)])
def test_train_epoch(name, opt_steps, puzzles):
    """
    One epoch of the production harness: deep supervision on every round
    for the single-run model, one optimizer step per detached segment for
    the recursion -- eight puzzles in batches of four, so two batches.
    """
    model = build(name)
    before = [value.detach().clone() for value in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    stats = training.train_epoch(model, puzzles, BUDGET, optimizer,
                                 np.random.default_rng(0),
                                 torch.device("cpu"))
    assert stats["opt_steps"] == opt_steps
    assert stats["checkpoints"] == 2 * (
        model.n_sup if model.segmented else model.rounds)
    assert np.isfinite(stats["loss"])
    assert finite(model)
    assert any(not torch.equal(one, other) for one, other
               in zip(before, model.parameters()))


def test_loss_decreases_on_a_fixed_batch(puzzles):
    """
    A smoke assertion beside the deterministic ones: the recursion does
    learn something on a batch it sees over and over.
    """
    model = build("trm")
    clues, target = tensors(puzzles)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(8):
        state = model.initial(clues)
        for _ in range(model.n_sup):
            state, logits = model.step(state)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.n), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state = state.detach()
            losses.append(loss.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


# --- the recursion: refresh, cycles and the detach boundary ----------------

def test_refresh_only_touches_the_answer(puzzles):
    """
    A cycle ends with one refresh of the answer from the latent state: it
    writes every copy of the answer loop and leaves every other port of
    the flat state exactly as it was.
    """
    model = build("trm")
    found = model.interaction
    clues, _ = tensors(puzzles)
    with torch.no_grad():
        state = model.initial(clues)
        refreshed = model.map.solver.refresh(found, state)
    assert refreshed.shape == state.shape == (BATCH, found.total)
    assert torch.isfinite(refreshed).all()
    assert not torch.equal(found.read(refreshed, model.answer),
                           found.read(state, model.answer))
    for key in (("cell", zoo.STATE), ("cell", zoo.CLUE)):
        assert torch.equal(found.read(refreshed, key),
                           found.read(state, key))
    every = found.read(refreshed, model.answer, every=True)
    assert every.shape[1] == 2 * model.n_cells
    assert torch.equal(every[:, 0::2], found.read(refreshed, model.answer))


def test_state_shapes_survive_every_cycle(puzzles):
    """ The state is a flat vector of the interaction's width, cycle after
    cycle, and stays finite. """
    model = build("trm")
    total = model.interaction.total
    clues, _ = tensors(puzzles)
    with torch.no_grad():
        state = model.initial(clues)
        for _ in range(4):
            state = model.map.solver.cycle(model.interaction, state)
            assert state.shape == (BATCH, total)
            assert torch.isfinite(state).all()
        for _ in range(N_SUP):
            state, logits = model.step(state)
            assert state.shape == (BATCH, total)
            assert logits.shape == (BATCH, 81, 9)


@pytest.mark.parametrize("cycles,differentiated", [(1, True), (2, False)])
def test_only_the_last_cycle_is_differentiated(cycles, differentiated,
                                               puzzles):
    """
    The detach boundary of the segmented recursion, pinned exactly: the
    first ``cycles - 1`` cycles run under ``no_grad``, so with more than
    one cycle no gradient reaches the state a step started from -- nor,
    therefore, the encoder that built it.
    """
    model = build("act")
    clues, target = tensors(puzzles)
    state = model.initial(clues).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    _, logits, _ = model.act_step(state, cycles=cycles)
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.n), target.reshape(-1)).backward()
    assert (state.grad is not None) == differentiated
    cell = model.map.ar["cell"]
    assert cell.encode[0].weight.grad is not None
    assert torch.isfinite(cell.encode[0].weight.grad).all()


def test_the_encoder_is_undifferentiated_under_act(puzzles):
    """
    The consequence the ACT trainer relies on: with more than one cycle
    per step the embedding and the learned initial answer receive no
    gradient, which is why the refilled states may be built under
    ``no_grad``.
    """
    model = build("act")
    clues, target = tensors(puzzles)
    model.zero_grad(set_to_none=True)
    _, logits, _ = model.act_step(model.initial(clues))
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.n), target.reshape(-1)).backward()
    assert model.embedding.weight.grad is None
    assert model.y0.grad is None
    assert model.readout.weight.grad is not None


# --- the halt head ---------------------------------------------------------

def test_halt_head_shapes(puzzles):
    """ The halt output is per cell for the soft-minimum head, the
    problem-level logit is its soft minimum, and both are finite. """
    model = build("act", halt_head="softmin")
    clues, target = tensors(puzzles)
    with torch.no_grad():
        _, logits, halt = model.act_step(model.initial(clues), grad=False)
        q = model.halt_logit(halt)
        loss = model.halt_loss(halt, logits, target)
    assert halt.shape == (BATCH, 81) and q.shape == (BATCH, )
    assert torch.isfinite(halt).all() and torch.isfinite(q).all()
    assert torch.isfinite(loss) and float(loss) >= 0.0
    assert float(q.max()) <= float(halt.min(dim=-1).values.max())

    pooled = build("act", halt_head="mean")
    with torch.no_grad():
        _, _, halt = pooled.act_step(pooled.initial(clues), grad=False)
    assert halt.shape == (BATCH, )
    assert torch.equal(pooled.halt_logit(halt), halt)


@pytest.mark.parametrize("halt_detach", [True, False])
def test_halt_detach_cuts_the_trunk(halt_detach, puzzles):
    """
    ``halt_detach`` says whether the head's loss trains the head alone.
    The head is initialised to zero weights, so the trunk would receive a
    zero gradient either way at that point: the weights are perturbed
    first, and then the difference is the detach and nothing else.
    """
    model = build("act", halt_detach=halt_detach)
    clues, target = tensors(puzzles)
    with torch.no_grad():
        model.map.solver.halt.weight.fill_(0.1)
    model.zero_grad(set_to_none=True)
    _, logits, halt = model.act_step(model.initial(clues))
    model.halt_loss(halt, logits, target).backward()
    head = model.map.solver.halt
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    trunk = model.map.solver.refresh.norm.weight.grad
    assert (trunk is None) == halt_detach
    if not halt_detach:
        assert float(trunk.abs().sum()) > 0.0


# --- the ACT training loop, with slot refill -------------------------------

def test_act_trainer_refills_its_slots(puzzles):
    """
    The slot-refill loop: one optimizer step per iteration, every slot
    advancing its own puzzle and refilled from the stream the moment it
    halts.  With a cap of two supervision steps every slot halts within
    two iterations, so the stream is drawn from more than once.
    """
    model = build("act")
    clues, target = tensors(puzzles)
    stream = training.ExampleStream(clues, target, np.random.default_rng(0))
    trainer = training.ACTTrainer(model, stream, batch_size=4)
    assert trainer.inputs.shape == (4, 81)
    assert trainer.state.shape == (4, model.interaction.total)
    assert stream.total_consumed() == 4

    before = [value.detach().clone() for value in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    stats = trainer.run(optimizer, scheduler, None, iterations=3)

    for key in ("loss", "ce", "q", "solved", "depth", "capped"):
        assert np.isfinite(stats[key]), key
    assert stats["halted"] > 0
    assert 1.0 <= stats["depth"] <= N_SUP
    assert stream.total_consumed() > 4
    assert torch.isfinite(trainer.state).all()
    assert bool((trainer.steps < model.n_sup).all())
    assert finite(model)
    assert any(not torch.equal(one, other) for one, other
               in zip(before, model.parameters()))


# --- the three evaluation protocols ----------------------------------------

def test_adaptive_evaluation_is_bounded(puzzles):
    """
    Early stopping under the paper's rule: every puzzle halts within the
    cap, and the reported depth is bounded by it.
    """
    model = build("act")
    scores = evaluations.evaluate_act(
        model, puzzles, max_steps=N_SUP, batch_size=4)
    assert 0.0 <= scores["cell"] <= 1.0 and 0.0 <= scores["board"] <= 1.0
    assert 1.0 <= scores["depth"] <= N_SUP


def test_selection_degenerates_to_act(puzzles):
    """
    With one rollout and no noise, the selected evaluation *is* the ACT
    evaluation, number for number -- which is what makes the difference at
    more rollouts attributable to the selection.
    """
    model = build("act", halt_head="softmin")
    plain = evaluations.evaluate_act(model, puzzles, max_steps=N_SUP,
                                     batch_size=4)
    selected = evaluations.evaluate_selected(
        model, puzzles, rollouts=1, sigma=0.0, max_steps=N_SUP,
        batch_size=4)
    for key in ("cell", "board", "depth"):
        assert selected[key] == plain[key], key
    assert selected["board_mean"] == plain["board"]
    assert selected["board_oracle"] == plain["board"]


def test_selection_is_bounded_and_deterministic(puzzles):
    model = build("act", halt_head="softmin")
    scores = [evaluations.evaluate_selected(
        model, puzzles, rollouts=3, sigma=0.5, max_steps=N_SUP,
        batch_size=4, seed=7) for _ in range(2)]
    assert scores[0] == scores[1]
    assert scores[0]["board_oracle"] >= scores[0]["board"] >= 0.0
    assert scores[0]["board_oracle"] >= scores[0]["board_mean"]


def test_noise_is_written_to_both_ends_of_the_loop(puzzles):
    """
    The two ports of a cell's answer loop intentionally carry the same
    ``y``, so one noise sample per cell is written to both ends rather
    than breaking the loop's consistency.
    """
    model = build("act")
    clues, _ = tensors(puzzles)
    found = model.interaction
    state = model.initial(clues)
    generator = torch.Generator().manual_seed(0)
    noisy = evaluations.perturb_answer(model, state, 0.5, generator)
    every = found.read(noisy, model.answer, every=True)
    assert torch.equal(every[:, 0::2], every[:, 1::2])
    assert torch.equal(evaluations.perturb_answer(model, state, 0.0), state)


# --- a checkpoint is the model ---------------------------------------------

def test_checkpoint_round_trip(puzzles):
    """
    Saving and loading a state dict reproduces the logits bitwise, which
    is what makes a trained artifact worth keeping.
    """
    trained = build("act")
    clues, _ = tensors(puzzles)
    optimizer = torch.optim.Adam(trained.parameters(), lr=1e-2)
    training.train_epoch(trained, puzzles, BUDGET, optimizer,
                         np.random.default_rng(0), torch.device("cpu"))
    stored = {key: value.clone()
              for key, value in trained.state_dict().items()}

    fresh = build("act", seed=1)
    with torch.no_grad():
        assert not torch.equal(fresh(clues), trained(clues))
    report = fresh.load_state_dict(stored, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    with torch.no_grad():
        assert torch.equal(fresh(clues), trained(clues))
        assert torch.equal(Decoder.decode(fresh(clues), clues),
                           Decoder.decode(trained(clues), clues))
