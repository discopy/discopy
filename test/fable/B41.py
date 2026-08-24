"""B41: coverage measures the test files themselves, no source is configured (pyproject.toml, tool.coverage.run).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_b41_coverage_scoped_to_the_package():
    with open(ROOT / 'pyproject.toml', 'rb') as stream:
        cfg = tomllib.load(stream)
    assert "source" in cfg["tool"]["coverage"]["run"]
