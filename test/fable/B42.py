"""B42: tests that cannot fail — str() asserts, an assert-free test and a swallowed readme block (test/cat.py:114, test/markov.py, test/readme.py:26).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from pathlib import Path

TEST = Path(__file__).parents[1]


def test_b42_cat_asserts_compare_before_str():
    src = (TEST / 'cat.py').read_text()
    assert "assert str(" not in src


def test_b42_markov_equations_actually_assert():
    src = (TEST / 'markov.py').read_text()
    body = src.split("def test_equations():", 1)[1].split("\ndef ", 1)[0]
    assert "assert" in body


def test_b42_readme_blocks_are_not_swallowed():
    src = (TEST / 'readme.py').read_text()
    assert "except SyntaxError:\n            continue" not in src
