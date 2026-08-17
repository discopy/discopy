import json

import polars as pl

from benchmark import report


def frame(*rows):
    return pl.DataFrame(
        rows,
        schema={
            "suite": pl.String,
            "family": pl.String,
            "case": pl.String,
            "size": pl.Int64,
            "median": pl.Float64,
        },
        orient="row",
    )


def write_run(path, *rows):
    path.write_text(json.dumps({"benchmarks": [{
        "group": [suite, family, case],
        "params": {"n": size},
        "stats": {"median": median},
    } for suite, family, case, size, median in rows]}))


def test_compare_uses_raw_same_runner_change():
    base = frame(
        ("suite", "family", "first", 1, 1.),
        ("suite", "family", "second", 1, 2.),
    )
    head = frame(
        ("suite", "family", "first", 1, 2.),
        ("suite", "family", "second", 1, 4.),
    )

    changes = report.compare(head, base)

    assert changes["change"].to_list() == [1., 1.]
    assert "normalised" not in changes.columns


def test_comparison_lists_only_important_changes():
    base = frame(
        ("suite", "family", "regressed", 1, 1.),
        ("suite", "family", "slow-boundary", 1, 1.),
        ("suite", "family", "faster", 1, 1.),
        ("suite", "family", "fast-boundary", 1, 1.),
        ("suite", "family", "base-only", 1, 1.),
    )
    head = frame(
        ("suite", "family", "regressed", 1, 1.5),
        ("suite", "family", "slow-boundary", 1, 1.25),
        ("suite", "family", "faster", 1, .5),
        ("suite", "family", "fast-boundary", 1, .75),
        ("suite", "family", "head-only", 1, 1.),
    )

    markdown = report.comparison_markdown(head, base, .25)

    assert "**FAIL:** 1 important regression found." in markdown
    assert "Found 1 important speedup among 4 shared measurements." in markdown
    assert "Head-only measurements: 1. Base-only measurements: 1." in markdown
    assert "| suite | family | regressed | 1 | 1 | 1.5 | +50.0% |" in markdown
    assert "| suite | family | faster | 1 | 1 | 0.5 | -50.0% |" in markdown
    assert "slow-boundary" not in markdown
    assert "fast-boundary" not in markdown


def test_main_writes_comparison_before_failing(tmp_path, monkeypatch):
    base, head = tmp_path / "base.json", tmp_path / "head.json"
    output = tmp_path / "output"
    write_run(base, ("suite", "family", "case", 1, 1.))
    write_run(head, ("suite", "family", "case", 1, 2.))
    monkeypatch.setattr(report, "write_reports", lambda *_: [])

    status = report.main([
        str(head), "--base", str(base), "--output", str(output)])

    assert status == 1
    assert "| suite | family | case | 1 | 1 | 2 | +100.0% |" in (
        output / "comparison.md").read_text()


def test_tied_speedups_are_sorted_by_key():
    base = frame(
        ("suite", "family", "second", 1, 1.),
        ("suite", "family", "first", 1, 1.),
    )
    head = frame(
        ("suite", "family", "second", 1, .5),
        ("suite", "family", "first", 1, .5),
    )

    markdown = report.comparison_markdown(head, base, .25)

    assert markdown.index("| suite | family | first |") < markdown.index(
        "| suite | family | second |")
