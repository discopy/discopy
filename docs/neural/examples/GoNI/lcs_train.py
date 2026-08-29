# -*- coding: utf-8 -*-

"""
Train the GoNI grid on the benchmark's ``lcs_length`` and report.

    python lcs_train.py

One run trains on the benchmark's 1000 training samples — 8 x 8 grids —
selects on the 32 validation samples at the same size, and reports the
selected model out of distribution on the 32 x 32 grids: the
benchmark's 32 test samples and the wider 128-sample split beside them.
The score is the benchmark's own for the ``b`` output, exact direction
per grid cell.  One evaluator instance owns the compiled maps, so the
one-off circuit build at 32 x 32 is shared across the seeds; the
weights are saved beside the report, so a later evaluation is a load,
not a rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import lcs_model as goni
from lcs_dataset import load

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def run(seed: int, epochs: int, batch_size: int, lr: float) -> tuple:
    torch.manual_seed(seed)
    splits = {split: load(split) for split in ("train", "val")}
    keys = torch.as_tensor(splits["train"]["keys"], dtype=torch.long)
    b = torch.as_tensor(splits["train"]["b"], dtype=torch.long)
    m, n = (int(splits["train"][name]) for name in ("x_len", "y_len"))
    grid = goni.Grid()
    optimizer = torch.optim.Adam(grid.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, best = [], None
    for epoch in range(epochs):
        order, losses = torch.randperm(len(keys), generator=generator), []
        for start in range(0, len(keys), batch_size):
            batch = order[start:start + batch_size]
            logits = grid(keys[batch], m, n)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), b[batch].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        entry = {"epoch": epoch,
                 "loss": sum(losses) / len(losses),
                 "val": goni.accuracy(grid, splits["val"])}
        history.append(entry)
        if best is None or entry["val"] >= best["val"]:
            best = dict(entry, state=[
                (name, tensor.detach().clone())
                for name, tensor in grid.state_dict().items()])
        print(f"epoch {epoch}: loss {entry['loss']:.4f} "
              f"val {entry['val']:.3f}", flush=True)
    grid.load_state_dict(dict(best.pop("state")))
    return {"seed": seed, "selected": best, "history": history}, grid


def main(seeds, epochs, batch_size, lr) -> dict:
    evaluator = goni.Grid()
    results = {}
    for seed in seeds:
        started = time.time()
        report, grid = run(seed, epochs, batch_size, lr)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        torch.save(grid.state_dict(), ARTIFACTS / f"lcs-seed{seed}.pt")
        evaluator.load_state_dict(grid.state_dict())
        for split in ("test", "wide"):
            report[split] = goni.accuracy(evaluator, load(split))
        report["minutes"] = (time.time() - started) / 60
        print(f"seed {seed}: val {report['selected']['val']:.3f} "
              f"test {report['test']:.3f} wide {report['wide']:.3f}",
              flush=True)
        results[seed] = report
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    arguments = parser.parse_args()
    results = main(arguments.seeds, arguments.epochs,
                   arguments.batch_size, arguments.lr)
    for seed, report in results.items():
        with open(ARTIFACTS / f"lcs-seed{seed}.json", "w") as stream:
            json.dump(report, stream, indent=2)
