# -*- coding: utf-8 -*-

"""
Render a ``pytest-benchmark`` JSON run as scaling tables and log-log plots,
with an optional regression gate against a committed baseline.

    python benchmark/report.py RUN.json [--output DIR]
                               [--baseline BASE.json] [--fail-threshold 0.25]

Reads the median CPU time of each ``(benchmark, family, case, size)`` from
``RUN.json``. It produces a hierarchical table as ``results.{html,md,csv}``
and one ``NAME-scaling.png`` per benchmark. With ``--baseline``, it joins the
two runs on all four keys, prints the per-cell deltas and exits non-zero if any
case regresses by more than ``--fail-threshold`` (a fraction, e.g. ``0.25`` =
25%).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import os

import polars as pl


def case_parts(label: str) -> tuple[str, str]:
    """Split a benchmark label into its case and canonical family."""
    case, family = label.rsplit(" (", 1)
    return case, family[:-1].lower().replace(" → ", "→")


def load(path: str) -> pl.DataFrame:
    """A tidy benchmark, family, case, size and median frame."""
    with open(path) as file:
        data = json.load(file)
    rows = []
    for bench in data["benchmarks"]:
        case, family = case_parts(bench.get("group") or bench["name"])
        rows.append({
            "benchmark": os.path.splitext(os.path.basename(bench[
                "fullname"].split("::", 1)[0]))[0].removeprefix("test_"),
            "family": family,
            "case": case,
            "size": int(bench["params"]["n"]),
            "median": float(bench["stats"]["median"]),
        })
    return pl.DataFrame(
        rows, schema={"benchmark": pl.String, "family": pl.String,
                      "case": pl.String, "size": pl.Int64,
                      "median": pl.Float64},
    ).sort("benchmark", "family", "case", "size")


def scaling_table(df: pl.DataFrame) -> pl.DataFrame:
    """One row per benchmark/family/case and one column per size."""
    ordered = pl.concat([
        df.filter(
            (pl.col("benchmark") == benchmark)
            & (pl.col("family") == family),
        ).sort("case", "size")
        for benchmark, spec in PLOTS.items()
        for families in spec.panels
        for family in families
    ])
    return ordered.pivot(
        on="size", index=["benchmark", "family", "case"], values="median",
    )


def to_markdown(table: pl.DataFrame) -> str:
    columns = table.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.iter_rows():
        cells = list(row[:3]) + [
            "" if value is None else f"{value:.4f}" for value in row[3:]]
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
    """Render benchmark and family as nested HTML row groups."""
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
        for depth in (1, 2):
            span = _rowspan(rows, i, depth)
            if span:
                cells.append(
                    f'<th scope="rowgroup" rowspan="{span}">'
                    f'{escape(str(row[depth - 1]))}</th>')
        cells.append(f'<th scope="row">{escape(str(row[2]))}</th>')
        cells += [
            f"<td>{'' if value is None else f'{value:.4f}'}</td>"
            for value in row[3:]]
        lines.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(lines + ["</tbody>", "</table>"])


@dataclass(frozen=True)
class Plot:
    """Declarative panel layout for one benchmark."""
    panels: tuple[tuple[str, ...], ...]
    figsize: tuple[int, int]
    title: str


PLOTS = {
    "composition": Plot(
        (("diagram", "hypergraph", "cmap"),), (19, 6),
        "Composition benchmark scaling (arXiv:2105.09257)"),
    "conversion": Plot((
        ("diagram→hypergraph", "hypergraph→cmap", "cmap→diagram"),
        ("hypergraph→diagram", "cmap→hypergraph", "diagram→cmap"),
    ), (19, 11), "Representation conversion benchmark scaling"),
}


REPRESENTATIONS = {
    "diagram": "Diagram",
    "hypergraph": "Hypergraph",
    "cmap": "CMap",
}


def family_title(family: str) -> str:
    """Human-readable title for a representation or conversion family."""
    return " → ".join(REPRESENTATIONS[name] for name in family.split("→"))


def case_colors(df: pl.DataFrame) -> dict[str, tuple]:
    """Stable colors shared by each case across every family and benchmark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted(df["case"].unique())
    palette = plt.get_cmap("tab20")
    return {case: palette(i % palette.N) for i, case in enumerate(cases)}


def plot(df: pl.DataFrame, path: str, spec: Plot, colors: dict) -> None:
    """Plot one benchmark according to its declarative panel layout."""
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
                     title=family_title(family))
            axis.legend(fontsize="small")
    for axis in axes.flat:
        axis.grid(True, which="both", linestyle=":", linewidth=.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("median CPU time (s)")
    figure.suptitle(spec.title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_report(df: pl.DataFrame, output: str) -> list[str]:
    """Write the hierarchical tables and one scaling plot per benchmark."""
    table = scaling_table(df)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        print(table)
    names = ["results.md", "results.csv", "results.html"]
    with open(os.path.join(output, names[0]), "w") as file:
        file.write(to_markdown(table) + "\n")
    table.write_csv(os.path.join(output, names[1]))
    with open(os.path.join(output, names[2]), "w") as file:
        file.write(to_html(table) + "\n")
    colors = case_colors(df)
    for (benchmark,), group in df.group_by("benchmark", maintain_order=True):
        name = f"{benchmark}-scaling.png"
        plot(group, os.path.join(output, name), PLOTS[benchmark], colors)
        names.append(name)
    return names


def compare(current: pl.DataFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """ Per-cell relative change vs baseline, worst first (shared only). """
    return current.join(
        baseline, on=["benchmark", "family", "case", "size"], suffix="_base",
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
    written = write_report(df, args.output)
    print(f"wrote {', '.join(written)} to {args.output}/")

    if not args.baseline:
        return 0
    if not os.path.exists(args.baseline):
        print(f"baseline {args.baseline} not found; skipping regression gate.")
        return 0
    deltas = compare(df, load(args.baseline))
    regressions = deltas.filter(pl.col("delta") > args.fail_threshold)
    with pl.Config(tbl_rows=-1):
        print(deltas.select(
            "benchmark", "family", "case", "size", "median",
            "median_base", "delta"))
    if len(regressions):
        print(f"REGRESSION: {len(regressions)} case(s) over "
              f"+{args.fail_threshold:.0%} vs baseline:")
        print(regressions.select(
            "benchmark", "family", "case", "size", "delta"))
        return 1
    print(f"no case regressed by more than +{args.fail_threshold:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
