# -*- coding: utf-8 -*-

"""
Train the GoNI matcher on the benchmark's ``kmp_matcher`` and report.

    python train.py --seed 0

One run trains on the benchmark's 1000 training samples at ``n = 16``,
selects on the 32 validation samples at the same size, and reports the
selected model out of distribution at ``n = 64``: the benchmark's 32
test samples and the wider 128-sample split beside them.  The artifact
is one json per seed with the whole history, so a table over seeds is a
read, not a rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import model as goni
from dataset import load

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def run(seed: int, epochs: int, batch_size: int, lr: float) -> dict:
    torch.manual_seed(seed)
    splits = {split: load(split) for split in
              ("train", "val", "test", "wide")}
    keys = torch.as_tensor(splits["train"]["keys"], dtype=torch.long)
    match = torch.as_tensor(splits["train"]["match"], dtype=torch.long)
    text, pattern = (int(splits["train"][name])
                     for name in ("text", "pattern"))
    matcher = goni.Matcher()
    optimizer = torch.optim.Adam(matcher.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, best = [], None
    for epoch in range(epochs):
        order, losses = torch.randperm(len(keys), generator=generator), []
        for start in range(0, len(keys), batch_size):
            batch = order[start:start + batch_size]
            scores = matcher(keys[batch], text, pattern)
            loss = torch.nn.functional.cross_entropy(scores, match[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        entry = {"epoch": epoch,
                 "loss": sum(losses) / len(losses),
                 "val": goni.accuracy(matcher, splits["val"])}
        history.append(entry)
        if best is None or entry["val"] >= best["val"]:
            best = dict(entry, state=[
                (name, tensor.detach().clone())
                for name, tensor in matcher.state_dict().items()])
        print(f"epoch {epoch}: loss {entry['loss']:.4f} "
              f"val {entry['val']:.3f}")
    matcher.load_state_dict(dict(best.pop("state")))
    report = {
        "seed": seed, "selected": best, "history": history,
        "test": goni.accuracy(matcher, splits["test"]),
        "wide": goni.accuracy(matcher, splits["wide"]),
    }
    print(f"seed {seed}: val {best['val']:.3f} "
          f"test {report['test']:.3f} wide {report['wide']:.3f}")
    return report, matcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    arguments = parser.parse_args()
    started = time.time()
    report, _ = run(arguments.seed, arguments.epochs,
                    arguments.batch_size, arguments.lr)
    report["minutes"] = (time.time() - started) / 60
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / f"kmp-seed{arguments.seed}.json", "w") as stream:
        json.dump(report, stream, indent=2)
