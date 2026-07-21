# -*- coding: utf-8 -*-

"""
Training, evaluation and diagnostics, shared by the three models.

The two supervision schemes live in :func:`train_epoch`: models A and B are
supervised on every round of one differentiated run, model C on every
supervision step of its segmented outer loop. Everything else -- optimizer,
clipping, batch size, decode rule, metrics -- is identical, so that a
difference in the results is a difference between the models.

Every run is cached under :data:`experiments.config.ARTIFACTS` as a
checkpoint holding the weights, the per-epoch history and the metadata of the
run, so re-running the notebook re-loads rather than re-trains.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from experiments import data as datasets
from experiments import models as zoo
from experiments.config import (
    ARTIFACTS, DEFAULT_LR, GRAD_CLIP, LR_GRID, MODELS, WIDTHS, Budget)

CE = torch.nn.functional.cross_entropy


def device_of(model) -> torch.device:
    """ The device the model lives on. """
    return next(model.parameters()).device


def seed_everything(seed: int) -> None:
    """ Fix every source of randomness we use. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def logits_of(model, clues, compute: int = None, deep: bool = False):
    """
    The digit logits of any of the three models under a common interface.

    Parameters:
        model : The solver.
        clues : The puzzles, of shape ``(batch, 81)``.
        compute : The test-time compute, i.e. rounds for A and B and
                  supervision steps for C, or ``None`` for the trained one.
        deep : Whether to return the logits of every supervised checkpoint.
    """
    if model.outer_loop:
        return model(clues, deep=deep, n_sup=compute)
    return model(clues, deep=deep, rounds=compute)


def decode(logits, clues):
    """ Argmax digits with the clues written back over the predictions. """
    predicted = logits.argmax(-1) + 1
    return torch.where(clues > 0, clues, predicted)


# --- evaluation -----------------------------------------------------------

@torch.no_grad()
def evaluate(model, split: datasets.Split, compute: int = None,
             batch_size: int = 256, buckets: bool = False) -> dict:
    """
    Cell and board accuracy on a split, optionally bucketed by givens.

    Parameters:
        model : The solver.
        split : The split to evaluate on.
        compute : The test-time compute, ``None`` for the trained one.
        batch_size : The evaluation batch size.
        buckets : Whether to also return the per-givens breakdown.
    """
    model.eval()
    device = device_of(model)
    correct, solved = [], []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        clues = torch.as_tensor(
            split.puzzles[start:stop], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split.solutions[start:stop], dtype=torch.long, device=device)
        matches = decode(logits_of(model, clues, compute), clues) == target
        correct.append(matches.float().mean(1).cpu().numpy())
        solved.append(matches.all(1).cpu().numpy())
    correct, solved = np.concatenate(correct), np.concatenate(solved)
    result = {"cell": float(correct.mean()), "board": float(solved.mean())}
    if buckets:
        givens = split.givens
        result["by_givens"] = pd.DataFrame({
            "givens": givens, "cell": correct, "board": solved.astype(float),
        }).groupby("givens").agg(
            cell=("cell", "mean"), board=("board", "mean"),
            n=("board", "size")).reset_index()
    return result


@torch.no_grad()
def sweep_compute(model, split: datasets.Split, values,
                  batch_size: int = 256) -> pd.DataFrame:
    """ Accuracy as a function of test-time compute. """
    return pd.DataFrame([
        dict(compute=value, **{
            key: score for key, score in
            evaluate(model, split, compute=value,
                     batch_size=batch_size).items()})
        for value in values])


@torch.no_grad()
def residual_curve(model, split: datasets.Split, steps: int = 32,
                   batch_size: int = 256) -> np.ndarray:
    """
    The relative step of the cell states along one run,
    ``||h_{t+1} - h_t|| / ||h_t||``, averaged over a batch of puzzles.

    For models A and B one step is one round of message passing; for model C
    it is one supervision step, the unit its outer loop advances by.
    """
    model.eval()
    device = device_of(model)
    clues = torch.as_tensor(
        split.puzzles[:batch_size], dtype=torch.long, device=device)
    states = []
    if model.outer_loop:
        state = model.initial(clues)
        for _ in range(steps):
            state, _ = model.step(state, grad=False)
            states.append(model.router.read(state, model.state_ports))
    else:
        init = model.initial(clues)
        for emitted in model.cells(init=init, n_rounds=steps, inject=True,
                                   return_rounds=True):
            flat = model.router(emitted)
            states.append(model.router.read(flat, model.state_ports)[
                ..., :model.widths.state_dim])
    stacked = torch.stack(states, 0).flatten(2)
    step = (stacked[1:] - stacked[:-1]).norm(dim=-1)
    return (step / stacked[:-1].norm(dim=-1).clamp_min(1e-9)
            ).mean(1).cpu().numpy()


@torch.no_grad()
def prediction_trace(model, puzzle, solution, steps) -> list[dict]:
    """
    One board's prediction at each of a list of compute values, for the
    qualitative panel: the decoded grid and the confidence of its argmax.
    """
    model.eval()
    device = device_of(model)
    clues = torch.as_tensor(puzzle, dtype=torch.long, device=device)[None]
    target = torch.as_tensor(solution, dtype=torch.long, device=device)[None]
    trace = []
    for value in steps:
        logits = logits_of(model, clues, compute=value)
        probability = logits.softmax(-1).max(-1).values[0].cpu().numpy()
        predicted = decode(logits, clues)
        trace.append({
            "compute": value,
            "grid": predicted[0].cpu().numpy().reshape(9, 9),
            "confidence": np.where(
                puzzle.reshape(9, 9) > 0, 1.0, probability.reshape(9, 9)),
            "correct": float((predicted == target).float().mean()),
            "solved": bool((predicted == target).all())})
    return trace


# --- training -------------------------------------------------------------

def batches(split: datasets.Split, budget: Budget, rng, device):
    """
    One shuffled epoch of ``(clues, target)`` pairs, with the sudoku
    symmetry group applied on the fly when the budget asks for it.
    """
    order = rng.permutation(len(split))
    for start in range(0, len(order), budget.batch_size):
        index = order[start:start + budget.batch_size]
        puzzles, solutions = split.puzzles[index], split.solutions[index]
        if budget.augment:
            puzzles, solutions = datasets.augment(puzzles, solutions, rng)
        yield (torch.as_tensor(puzzles, dtype=torch.long, device=device),
               torch.as_tensor(solutions, dtype=torch.long,
                               device=device) - 1)


def train_epoch(model, split, budget, optimizer, rng, device) -> dict:
    """
    One epoch under the model's own deep supervision.

    Models A and B average the cross-entropy over every round of a single
    backward graph, so one batch is one optimizer step. Model C is supervised
    once per detached segment of its outer loop, so one batch is ``n_sup``
    optimizer steps -- the residual asymmetry between the two schemes, which
    we report rather than hide.

    Returns the mean loss *per supervised checkpoint* for both schemes, so
    that the two losses are on the same scale.
    """
    model.train()
    total, checkpoints, steps = 0.0, 0, 0
    for clues, target in batches(split, budget, rng, device):
        flat = target.reshape(-1)
        if model.outer_loop:
            state = model.initial(clues)
            for _ in range(model.n_sup):
                state, logits = model.step(state)
                loss = CE(logits.reshape(-1, model.n), flat)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                state, steps = state.detach(), steps + 1
                total, checkpoints = total + loss.item(), checkpoints + 1
        else:
            every = model(clues, deep=True)
            losses = [CE(logits.reshape(-1, model.n), flat)
                      for logits in every]
            loss = sum(losses) / len(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            steps += 1
            total += sum(item.item() for item in losses)
            checkpoints += len(losses)
    return {"loss": total / max(checkpoints, 1), "opt_steps": steps,
            "checkpoints": checkpoints}


def checkpoint_path(name: str, budget: Budget, seed: int):
    """ Where a run's artifacts live. """
    return ARTIFACTS / f"{budget.name}-{name}-seed{seed}.pt"


def train_model(name: str, budget: Budget, seed: int, train_split,
                valid_split, lr: float = None, device=None,
                resume: bool = True, log=print):
    """
    Train one model with one seed, or load it back when it is already cached.

    Parameters:
        name : ``"goi"``, ``"rrn"`` or ``"trm"``.
        budget : The budget giving data size, epochs, depth and batch size.
        seed : The seed, fixed before the model is built so that the
               initialisation is reproducible too.
        train_split : The training puzzles, already subsampled.
        valid_split : The split evaluated after every epoch.
        lr : The learning rate, from :data:`DEFAULT_LR` by default.
        resume : Whether to load a cached checkpoint if one exists.
        log : Where to print progress.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    lr = DEFAULT_LR[name] if lr is None else lr
    path = checkpoint_path(name, budget, seed)

    seed_everything(seed)
    model = zoo.build(name, budget).to(device)
    if resume and path.exists():
        stored = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(stored["state_dict"])
        return model, pd.DataFrame(stored["history"]), stored["meta"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, start = [], time.perf_counter()
    for epoch in range(budget.epochs):
        tick = time.perf_counter()
        stats = train_epoch(model, train_split, budget, optimizer, rng, device)
        scores = evaluate(model, valid_split)
        history.append({
            "epoch": epoch + 1, "seconds": time.perf_counter() - tick,
            **stats, **scores})
        log(f"  {name} seed {seed} epoch {epoch + 1:2d}/{budget.epochs}  "
            f"loss {stats['loss']:.4f}  cell {scores['cell']:.4f}  "
            f"board {scores['board']:.4f}  ({history[-1]['seconds']:.0f}s)")
    meta = {
        "model": name, "seed": seed, "lr": lr, "budget": asdict(budget),
        "widths": WIDTHS[name].asdict(),
        "parameters": zoo.count_parameters(model),
        "wires": model.n_wires, "boxes": len(model.grid.boxes),
        "seconds": time.perf_counter() - start,
        "seconds_per_epoch": float(np.mean([h["seconds"] for h in history])),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.type == "cuda" else float("nan")),
        "opt_steps_per_epoch": history[-1]["opt_steps"],
        "checkpoints_per_epoch": history[-1]["checkpoints"]}
    torch.save({"state_dict": model.state_dict(), "history": history,
                "meta": meta}, path)
    return model, pd.DataFrame(history), meta


def lr_search(name: str, budget: Budget, train_split, valid_split,
              grid=LR_GRID, device=None, log=print) -> pd.DataFrame:
    """
    A small learning-rate grid on the validation split, cached to disk.

    One seed, a reduced number of epochs and a reduced training subsample:
    enough to rule out a badly scaled learning rate without spending the
    experiment budget on tuning.
    """
    path = ARTIFACTS / f"{budget.name}-lr-{name}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    from dataclasses import replace
    # the epochs go in the name: a cached search run must never be reused
    # after `lr_epochs` changes, or a 1-epoch result would be read back as if
    # it had been trained for the new number of epochs.
    small = replace(budget, epochs=budget.lr_epochs,
                    name=f"{budget.name}-lrsearch{budget.lr_epochs}e-{name}")
    subsample = train_split.subsample(budget.lr_n_train)
    rows = []
    for lr in grid:
        _, history, _ = train_model(
            name, replace(small, name=f"{small.name}-{lr:g}"), 0, subsample,
            valid_split, lr=lr, device=device, log=lambda *a: None)
        rows.append({"model": name, "lr": lr,
                     "cell": history["cell"].iloc[-1],
                     "board": history["board"].iloc[-1],
                     "loss": history["loss"].iloc[-1]})
        log(f"  {name} lr {lr:g}: cell {rows[-1]['cell']:.4f}  "
            f"board {rows[-1]['board']:.4f}")
    path.write_text(json.dumps(rows, indent=2))
    return pd.DataFrame(rows)


def run_all(budget: Budget, splits, lrs=None, device=None, log=print,
            models=None):
    """
    Train every model with every seed, returning the trained models, the
    per-epoch histories and the run metadata.

    Parameters:
        budget : The budget to run.
        splits : The output of :func:`experiments.data.load`, subsampled.
        lrs : The learning rate per model, from a grid search or the default.
        models : Which models to train, all three by default. Runs are cached
                 per ``(budget, model, seed)``, so two processes given
                 disjoint model sets can share the work across two GPUs.
    """
    lrs = lrs or DEFAULT_LR
    trained, histories, metas = {}, [], []
    for name in (models or MODELS):
        for seed in budget.seeds:
            log(f"{name} seed {seed} (lr {lrs[name]:g})")
            model, history, meta = train_model(
                name, budget, seed, splits["train"], splits["valid"],
                lr=lrs[name], device=device, log=log)
            trained[(name, seed)] = model
            histories.append(history.assign(model=name, seed=seed))
            metas.append(meta)
    return trained, pd.concat(histories, ignore_index=True), metas
