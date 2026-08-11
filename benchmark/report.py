# -*- coding: utf-8 -*-

"""
Render a ``pytest-benchmark`` JSON run as scaling tables and log-log plots,
with an optional regression gate against a committed baseline.

    python benchmark/report.py RUN.json [--output DIR]
                               [--baseline BASE.json] [--fail-threshold 0.25]

Reads the median CPU time of each ``(suite, family, case, size)`` from
``RUN.json``. For each suite ``NAME``, it produces a hierarchical table as
``NAME-results.{html,md,csv}`` and a ``NAME-scaling.png`` plot. With
``--baseline``, it joins the two runs on all four keys, prints the per-cell
deltas and exits non-zero if any case regresses by more than
``--fail-threshold`` (a fraction, e.g. ``0.25`` = 25%).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import gzip
import json
import os

import polars as pl


def load(path: str, opener=open) -> pl.DataFrame:
    """A tidy suite, family, case, size and median frame."""
    with opener(path, "rt") as file:
        data = json.load(file)
    rows = []
    for bench in data["benchmarks"]:
        suite, family, case = bench["group"]
        rows.append({
            "suite": suite,
            "family": family,
            "case": case,
            "size": int(bench["params"]["n"]),
            "median": float(bench["stats"]["median"]),
        })
    return pl.DataFrame(
        rows, schema={"suite": pl.String, "family": pl.String,
                      "case": pl.String, "size": pl.Int64,
                      "median": pl.Float64},
    ).sort("suite", "family", "case", "size")


def scaling_table(df: pl.DataFrame, spec: Plot) -> pl.DataFrame:
    """One row per family/case and one column per size."""
    ordered = pl.concat([
        df.filter(pl.col("family") == family).sort("case", "size")
        for families in spec.panels
        for family in families
    ])
    table = ordered.pivot(
        on="size", index=["family", "case"], values="median",
    )
    return table.select(
        "family", "case", *sorted(table.columns[2:], key=int))


def to_markdown(table: pl.DataFrame) -> str:
    columns = table.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.iter_rows():
        cells = list(row[:2]) + [
            "" if value is None else f"{value:.4f}" for value in row[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rowspan(rows: list[tuple], i: int, depth: int) -> int:
    """Length of a row group, or zero when ``i`` is not its first row."""
    key = rows[i][:depth]
    if i and rows[i - 1][:depth] == key:
        return 0
    return next((j - i for j in range(i + 1, len(rows))
                 if rows[j][:depth] != key), len(rows) - i)


def to_html(table: pl.DataFrame) -> str:
    """Render family as an HTML row group."""
    rows, columns = list(table.iter_rows()), table.columns
    lines = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        "<title>Benchmark scaling</title>",
        "<style>table{border-collapse:collapse}th,td{border:1px solid #bbb;"
        "padding:.3rem .5rem;text-align:right}th[scope=rowgroup],"
        "th[scope=row]{text-align:left;vertical-align:top}</style>",
        "<table>",
        "<thead><tr>" + "".join(
            f'<th scope="col">{escape(column)}</th>' for column in columns)
        + "</tr></thead>",
        "<tbody>",
    ]
    for i, row in enumerate(rows):
        cells = []
        span = _rowspan(rows, i, 1)
        if span:
            cells.append(
                f'<th scope="rowgroup" rowspan="{span}">'
                f'{escape(str(row[0]))}</th>')
        cells.append(f'<th scope="row">{escape(str(row[1]))}</th>')
        cells += [
            f"<td>{'' if value is None else f'{value:.4f}'}</td>"
            for value in row[2:]]
        lines.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(lines + ["</tbody>", "</table>"])


@dataclass(frozen=True)
class Plot:
    """Declarative panel layout for one suite."""
    panels: tuple[tuple[str, ...], ...]
    figsize: tuple[int, int]
    title: str


PLOTS = {
    "composition": Plot(
        (("Diagram", "Hypergraph", "CMap"),), (19, 6),
        "Composition benchmark scaling (arXiv:2105.09257)"),
    "conversion": Plot((
        ("Diagram → Hypergraph", "Hypergraph → CMap", "CMap → Diagram"),
        ("Hypergraph → Diagram", "CMap → Hypergraph", "Diagram → CMap"),
    ), (19, 11), "Representation conversion benchmark scaling"),
}


def case_colors(df: pl.DataFrame) -> dict[str, tuple]:
    """Stable colors shared by each case across every family and suite."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted(df["case"].unique())
    palette = plt.get_cmap("tab20")
    return {case: palette(i % palette.N) for i, case in enumerate(cases)}


def plot(df: pl.DataFrame, path: str, spec: Plot, colors: dict) -> None:
    """Plot one suite according to its declarative panel layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(spec.panels), len(spec.panels[0]), figsize=spec.figsize,
        sharey=True, squeeze=False)
    for row, families in enumerate(spec.panels):
        for column, family in enumerate(families):
            axis = axes[row, column]
            panel = df.filter(pl.col("family") == family)
            for (case,), group in panel.group_by("case", maintain_order=True):
                ordered = group.sort("size")
                axis.plot(
                    ordered["size"].to_list(), ordered["median"].to_list(),
                    marker="o", label=case, color=colors[case])
            axis.set(xscale="log", yscale="log", xlabel="size $n$",
                     title=family)
            axis.legend(fontsize="small")
    for axis in axes.flat:
        axis.grid(True, which="both", linestyle=":", linewidth=.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("median CPU time (s)")
    figure.suptitle(spec.title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_reports(df: pl.DataFrame, output: str) -> list[str]:
    """Write one hierarchical table and scaling plot per suite."""
    colors = case_colors(df)
    names = []
    for suite, spec in PLOTS.items():
        group = df.filter(pl.col("suite") == suite)
        if group.is_empty():
            continue
        table = scaling_table(group, spec)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
            print(table)
        report_names = [
            f"{suite}-results.{extension}"
            for extension in ("md", "csv", "html")]
        with open(os.path.join(output, report_names[0]), "w") as file:
            file.write(to_markdown(table) + "\n")
        table.write_csv(os.path.join(output, report_names[1]))
        with open(os.path.join(output, report_names[2]), "w") as file:
            file.write(to_html(table) + "\n")
        plot_name = f"{suite}-scaling.png"
        plot(group, os.path.join(output, plot_name), spec, colors)
        names += report_names + [plot_name]
    return names


def compare(current: pl.DataFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """ Per-cell relative change vs baseline, worst first (shared only).

    ``delta`` is the raw change in median and ``normalised`` divides it by the
    run-wide median change, i.e. by the speed of the machine the run landed on.
    GitHub hands out several CPU models for the same runner label, so a raw
    delta mixes the machine in with the code. Writing a measurement as
    ``t = machine * code * baseline``, the median over cases estimates the
    machine -- most cases are unchanged, and the median ignores the few that
    are not -- and dividing it out leaves the change due to the code alone.
    Dividing rather than subtracting keeps the threshold meaning the same
    thing on every machine: subtracting would scale it by the machine factor.
    """
    return current.join(
        baseline, on=["suite", "family", "case", "size"], suffix="_base",
    ).with_columns(
        ((pl.col("median") - pl.col("median_base")) / pl.col("median_base"))
        .alias("delta"),
    ).with_columns(
        ((1 + pl.col("delta")) / (1 + pl.col("delta").median()) - 1)
        .alias("normalised"),
    ).sort("normalised", descending=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a pytest-benchmark run; optionally gate.")
    parser.add_argument("run", help="pytest-benchmark --benchmark-json file")
    parser.add_argument("--output", default="benchmark-results")
    parser.add_argument(
        "--baseline", help="baseline --benchmark-json to gate against")
    parser.add_argument(
        "--fail-threshold", type=float, default=0.25,
        help="fail if a case regresses by more than this fraction relative to"
             " the run-wide median change")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load(args.run)
    written = write_reports(df, args.output)
    print(f"wrote {', '.join(written)} to {args.output}/")

    if not args.baseline:
        return 0
    if not os.path.exists(args.baseline):
        print(f"baseline {args.baseline} not found; skipping regression gate.")
        return 0
    deltas = compare(df, load(args.baseline, gzip.open))
    machine = deltas["delta"].median()
    regressions = deltas.filter(
        pl.col("normalised") > args.fail_threshold)
    with pl.Config(tbl_rows=-1):
        print(deltas.select(
            "suite", "family", "case", "size", "median",
            "median_base", "delta", "normalised"))
    print(f"the machine this run landed on is {machine:+.1%} "
          "off the baseline's, measured as the run-wide median delta.")
    if len(regressions):
        print(f"REGRESSION: {len(regressions)} case(s) over "
              f"+{args.fail_threshold:.0%} normalised vs baseline:")
        print(regressions.select(
            "suite", "family", "case", "size", "delta", "normalised"))
        return 1
    print(f"no case regressed by more than +{args.fail_threshold:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
