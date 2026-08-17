# -*- coding: utf-8 -*-

"""
The test-split report for a search winner: accuracy against test-time
compute, then the same under answer noise with the halt head selecting
among rollouts.

    CUDA_VISIBLE_DEVICES=1 python eval_best.py \\
        ../../artifacts/optuna-act-extreme-trial27.pt

The search never touches ``test`` -- it ranks trials on 2,000 puzzles of
``valid``, with checkpoint selection over the evaluations of a trial and
then trial selection over the study, so its number is a doubly-selected
maximum on a small sample.  This script is what turns that into a claim:
one pass over the authors' 422,786-puzzle test split at each depth, and a
noise grid on a subsample, both written to JSON as they go so a long run
reports before it finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
import evaluate as evaluations  # noqa: E402
from config import ARTIFACTS, EXTREME_TRM  # noqa: E402


class Recorder:
    """
    Every accuracy the report measures, one row per ``(stage, compute,
    sigma, protocol, metric)``, rewritten after each row lands.

    A sweep's own JSON keeps its nesting and its ``npz`` keeps the
    per-example arrays; this is the flat view a plot reads directly and a
    hand can edit, so a figure drawn weeks later needs neither those files
    nor this script.  Rows accumulate across the stages of one invocation
    and :meth:`merge` puts several invocations into a single table.
    """

    #: A tidy table: the keys that identify a measurement, then its value.
    FIELDS = ("stage", "split", "n_puzzles", "compute", "sigma", "rollouts",
              "survivors", "protocol", "metric", "value", "depth", "seconds",
              "checkpoint")

    #: ``summarize`` names, as the metric column spells them.  ``single``
    #: is the mean over rollouts, so at one rollout it is just the board.
    METRICS = (("single", "board"), ("single_cell", "cell"),
               ("best_of_k", "best_of_k"), ("pass_at_k", "pass_at_k"))

    def __init__(self, stem: str, checkpoint: str = ""):
        self.rows: list = []
        self.stem, self.checkpoint = stem, checkpoint

    @property
    def paths(self) -> tuple:
        """ The CSV to plot from and the JSON to read back. """
        return (ARTIFACTS / f"{self.stem}-records.csv",
                ARTIFACTS / f"{self.stem}-records.json")

    def add(self, **row) -> None:
        """ One measurement, written through to disk immediately. """
        row.setdefault("split", "test")
        row.setdefault("checkpoint", self.checkpoint)
        self.rows.append({field: row.get(field) for field in self.FIELDS})
        self.flush()

    def flush(self) -> None:
        """ The table as it stands, as CSV and as JSON. """
        table, records = self.paths
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        records.write_text(json.dumps(self.rows, indent=1))

    def sweep(self, payload: dict, stage: str, n_puzzles: int) -> None:
        """
        The ``(cap, sigma)`` boxes of a :func:`evaluate.noise_sweep`,
        flattened: both protocols and every metric, so nothing measured is
        reachable only through the ``npz``.
        """
        settings = payload["protocol"]
        for cap in payload["caps"]:
            for sigma in payload["sigmas"]:
                box = payload["results"][cap][sigma]
                for protocol in ("fixed", "act"):
                    block = box[protocol]
                    for key, metric in self.METRICS:
                        self.add(stage=stage, n_puzzles=n_puzzles,
                                 compute=cap, sigma=sigma,
                                 rollouts=settings["rollouts"],
                                 survivors=settings["survivors"],
                                 protocol=protocol, metric=metric,
                                 value=block[key], depth=block.get("depth"))

    @classmethod
    def merge(cls, stems, stem: str) -> tuple:
        """
        Several runs' tables as one, for the stages that ran as separate
        jobs.  Rows carry the split size they were measured on, so a plot
        can keep the anchor and the subsample apart.
        """
        merged = cls(stem)
        for source in stems:
            path = ARTIFACTS / f"{source}-records.json"
            if path.exists():
                merged.rows.extend(json.loads(path.read_text()))
        merged.flush()
        return merged.paths


def fixed_sweep(model, split, computes, batch_size, path, recorder=None,
                log=print) -> list:
    """
    Board and cell accuracy at each depth, appended to ``path`` as each
    one lands: a pass at depth ``T`` costs ``T``, so the sweep to 512
    costs thirty-two times its first row and should not be all-or-nothing.
    """
    rows = []
    for compute in computes:
        tick = time.perf_counter()
        scores = evaluations.evaluate(
            model, split, compute=compute, batch_size=batch_size)
        rows.append({"compute": compute, "seconds": time.perf_counter() - tick,
                     **scores})
        log(f"  T={compute:4d}  board {scores['board']:.4f}  "
            f"cell {scores['cell']:.4f}  ({rows[-1]['seconds'] / 60:.1f} min)")
        path.write_text(json.dumps(rows, indent=1))
        if recorder is not None:
            for metric in ("board", "cell"):
                recorder.add(stage="fixed", n_puzzles=len(split),
                             compute=compute, sigma=0.0, rollouts=1,
                             survivors=1, protocol="fixed", metric=metric,
                             value=scores[metric],
                             seconds=rows[-1]["seconds"])
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--merge", nargs="+", metavar="STEM",
                        help="combine the record tables of these stems into "
                             "--stem and exit, for stages that ran as "
                             "separate jobs")
    parser.add_argument("--computes", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--n-fixed", type=int, default=0,
                        help="test puzzles for the fixed sweep, 0 for all")
    parser.add_argument("--n-noise", type=int, default=2000,
                        help="test puzzles for the noise grid, which costs "
                             "--rollouts trajectories per puzzle")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--rollouts", type=int, default=32,
                        help="noise paths per puzzle; the recorded study "
                             "used 32 with a beam of 4")
    parser.add_argument("--survivors", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--slice", default=None,
                        help="A:B, a contiguous slice of the test split, so "
                             "the same job runs sharded over several GPUs; "
                             "every puzzle is scored independently, so the "
                             "shards concatenate")
    parser.add_argument("--noiseless", action="store_true",
                        help="also sweep with one noiseless path, the "
                             "independent baseline the noisy grid is read "
                             "against; one trajectory records every cap, so "
                             "it costs one deep run rather than a pass each")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.merge:
        stem = arguments.stem or "testeval-merged"
        for path in Recorder.merge(arguments.merge, stem):
            print(f"wrote {path}", flush=True)
        return 0
    if arguments.checkpoint is None:
        parser.error("a checkpoint is required unless --merge is given")

    torch.set_float32_matmul_precision("high")
    model, meta = evaluations.load(
        arguments.checkpoint, "act", widths=EXTREME_TRM["widths"])
    model.map.compile_rounds(mode="reduce-overhead")
    splits = dataset.load_extreme("special_large", verify=False)
    test = splits["test"]
    stem = arguments.stem or f"testeval-{arguments.checkpoint.stem}"
    if arguments.slice:
        start, stop = (int(part) for part in arguments.slice.split(":"))
        test = dataset.Split(test.name, test.puzzles[start:stop],
                             test.solutions[start:stop], test.surrogate)
        stem = f"{stem}-{start}_{stop}"
        print(f"slice [{start}:{stop}] of the test split", flush=True)

    print(f"{arguments.checkpoint.name}: valid board on the search's own "
          f"2,000 puzzles was {meta.get('valid_board')}", flush=True)
    print(f"test split: {len(test):,} puzzles", flush=True)
    recorder = Recorder(stem, arguments.checkpoint.name)

    if not arguments.skip_fixed:
        split = test if not arguments.n_fixed \
            else test.subsample(arguments.n_fixed)
        print(f"\nfixed compute, {len(split):,} test puzzles:", flush=True)
        fixed_sweep(model, split, arguments.computes, arguments.batch_size,
                    ARTIFACTS / f"{stem}-fixed.json", recorder=recorder,
                    log=lambda line: print(line, flush=True))

    if arguments.noiseless:
        split = test.subsample(arguments.n_noise)
        print(f"\nnoiseless, {len(split):,} test puzzles, one path:",
              flush=True)
        clean = evaluations.noise_sweep(
            model, split, caps=arguments.computes, sigmas=(0.0, ),
            rollouts=1, survivors=1, batch_size=arguments.batch_size,
            log=lambda *a, **kw: print("  ", *a, flush=True))
        print(evaluations.table(clean["results"], clean["caps"],
                                clean["sigmas"], "fixed", "single"))
        recorder.sweep(clean, "clean", len(split))
        evaluations.write_sweep(clean, f"{stem}-clean", meta)

    if not arguments.skip_noise:
        split = test.subsample(arguments.n_noise)
        print(f"\nnoise + halt-head selection, {len(split):,} test puzzles, "
              f"{arguments.rollouts} paths per puzzle, beam "
              f"{arguments.survivors}:", flush=True)
        payload = evaluations.noise_sweep(
            model, split, caps=arguments.computes, sigmas=arguments.sigmas,
            rollouts=arguments.rollouts, survivors=arguments.survivors,
            batch_size=arguments.batch_size,
            log=lambda *a, **kw: print("  ", *a, flush=True))
        recorder.sweep(payload, "noise", len(split))
        print("\nboard, fixed compute, best-of-k selected by the halt logit:")
        print(evaluations.table(payload["results"], payload["caps"],
                                payload["sigmas"], "fixed", "best_of_k"))
        print("\nboard, single rollout (the mean of one noisy path):")
        print(evaluations.table(payload["results"], payload["caps"],
                                payload["sigmas"], "fixed", "single"))
        print("\nboard, pass@k -- what a perfect verifier would reach:")
        print(evaluations.table(payload["results"], payload["caps"],
                                payload["sigmas"], "fixed", "pass_at_k"))
        print("\nwrote " + ", ".join(
            str(p) for p in evaluations.write_sweep(payload, stem, meta)))
    print("\nrecords: " + ", ".join(str(p) for p in recorder.paths),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
