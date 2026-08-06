# -*- coding: utf-8 -*-

"""Tests for the benchmark report schema and hierarchical HTML table."""

import polars as pl

from benchmark.report import case_parts, scaling_table, to_html


def test_case_parts():
    assert case_parts("tensor (Diagram)") == ("tensor", "diagram")
    assert case_parts("tensor (Diagram → CMap)") == (
        "tensor", "diagram→cmap")


def test_hierarchical_table():
    df = pl.DataFrame({
        "benchmark": ["composition", "composition", "composition",
                      "conversion"],
        "family": ["diagram", "diagram", "hypergraph", "diagram→cmap"],
        "case": ["series", "tensor", "series", "series"],
        "size": [10, 10, 10, 10],
        "median": [1., 2., 3., 4.],
    })
    table = scaling_table(df)
    assert table.columns == ["benchmark", "family", "case", "10"]
    assert table.select("benchmark", "family", "case").rows() == [
        ("composition", "diagram", "series"),
        ("composition", "diagram", "tensor"),
        ("composition", "hypergraph", "series"),
        ("conversion", "diagram→cmap", "series"),
    ]
    html = to_html(table)
    assert '<th scope="rowgroup" rowspan="3">composition</th>' in html
    assert '<th scope="rowgroup" rowspan="2">diagram</th>' in html
    assert '<th scope="rowgroup" rowspan="1">conversion</th>' in html
