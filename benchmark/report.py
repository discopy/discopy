# -*- coding: utf-8 -*-

"""
Render a ``pytest-benchmark`` JSON run as a scaling table + log-log plot,
with an optional regression gate against a committed baseline.

    python benchmark/report.py RUN.json [--output DIR]
                               [--baseline BASE.json] [--fail-threshold 0.25]

Reads the median CPU time of each ``(case, size)`` from ``RUN.json`` and
writes ``results.md``, ``results.csv`` and ``scaling.png`` into ``DIR``
(default ``benchmark-results``). With ``--baseline``, joins the two runs on
``(case, size)`` in polars, prints the per-cell deltas, and exits non-zero
if any case regresses by more than ``--fail-threshold`` (a fraction, e.g.
``0.25`` = 25%).
"""
from __future__ import annotations

import argparse
import json
import os

import polars as pl


def load(path: str) -> pl.DataFrame:
    """ A tidy ``(case, n, median)`` frame from a pytest-benchmark run. """
    with open(path) as file:
        data = json.load(file)
    rows = [
        {
            "case": bench.get("group") or bench["name"],
            "n": int(bench["params"]["n"]),
            "median": float(bench["stats"]["median"]),
        }
        for bench in data["benchmarks"]
    ]
    return pl.DataFrame(
        rows, schema={"case": pl.String, "n": pl.Int64, "median": pl.Float64},
    ).sort("case", "n")


def scaling_table(df: pl.DataFrame) -> pl.DataFrame:
    """ One row per case, one column per size, holding the median seconds. """
    return df.sort("n").pivot(
        on="n", index="case", values="median").sort("case")


def to_markdown(table: pl.DataFrame) -> str:
    columns = table.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.iter_rows():
        cells = [row[0]] + ["" if v is None else f"{v:.4f}" for v in row[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def benchmark_family(name: str) -> str:
    """ Workload shared by analogous representation-specific cases. """
    name = name.rsplit(" (", 1)[0]
    return {
        "adder evaluation": "adder functor",
        "staircase foliation": "staircase",
        "staircase to hypergraph": "staircase",
        "staircase to cmap": "staircase",
        "staircase evaluation": "staircase",
        "spiral normal_form": "spiral equality",
        "spiral evaluation": "spiral equality",
        "transpose snake removal": "transpose equality",
        "transpose evaluation": "transpose equality",
    }.get(name, name)


def plot(df: pl.DataFrame, path: str) -> None:
    """ Log-log scaling plot, grouped into one panel per representation. """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["Diagram", "Hypergraph", "CMap", "Tensor"]
    figure, grid = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    axes = grid.flatten()
    families = sorted({benchmark_family(name) for name in df["case"]})
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i % 10) for i, name in enumerate(families)}
    for (name,), group in df.group_by("case", maintain_order=True):
        panel = next(
            i for i, title in enumerate(panels) if f"({title})" in name)
        axis = axes[panel]
        ordered = group.sort("n")
        axis.plot(ordered["n"].to_list(), ordered["median"].to_list(),
                  marker="o", label=name, color=colors[benchmark_family(name)])
    for axis, title in zip(axes, panels):
        axis.set(xscale="log", yscale="log", xlabel="size $n$", title=title)
        axis.grid(True, which="both", linestyle=":", linewidth=.5)
        axis.legend(fontsize="small")
    axes[0].set_ylabel("median CPU time (s)")
    figure.suptitle("Composition and tensor evaluation benchmark scaling")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def compare(current: pl.DataFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """ Per-cell relative change vs baseline, worst first (shared only). """
    return current.join(
        baseline, on=["case", "n"], suffix="_base",
    ).with_columns(
        ((pl.col("median") - pl.col("median_base")) / pl.col("median_base"))
        .alias("delta"),
    ).sort("delta", descending=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a pytest-benchmark run; optionally gate.")
    parser.add_argument("run", help="pytest-benchmark --benchmark-json file")
    parser.add_argument("--output", default="benchmark-results")
    parser.add_argument(
        "--baseline", help="baseline --benchmark-json to gate against")
    parser.add_argument(
        "--fail-threshold", type=float, default=0.25,
        help="fail if a case's median regresses by more than this fraction")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load(args.run)
    table = scaling_table(df)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        print(table)
    with open(os.path.join(args.output, "results.md"), "w") as file:
        file.write(to_markdown(table) + "\n")
    table.write_csv(os.path.join(args.output, "results.csv"))
    plot(df, os.path.join(args.output, "scaling.png"))
    print(f"wrote results.md, results.csv, scaling.png to {args.output}/")

    if not args.baseline:
        return 0
    if not os.path.exists(args.baseline):
        print(f"baseline {args.baseline} not found; skipping regression gate.")
        return 0
    deltas = compare(df, load(args.baseline))
    regressions = deltas.filter(pl.col("delta") > args.fail_threshold)
    with pl.Config(tbl_rows=-1):
        print(deltas.select("case", "n", "median", "median_base", "delta"))
    if len(regressions):
        print(f"REGRESSION: {len(regressions)} case(s) over "
              f"+{args.fail_threshold:.0%} vs baseline:")
        print(regressions.select("case", "n", "delta"))
        return 1
    print(f"no case regressed by more than +{args.fail_threshold:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
