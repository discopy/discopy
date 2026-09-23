"""Tests for the pylint plugin reading an ``@axiom`` law as a classmethod."""

import pathlib
import subprocess
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"

LAW = '''
def axiom(equation):
    return equation


class Category:
    attribute = 1

    @axiom
    def law(cls, x):
        return cls.attribute
'''


def messages(path, *options):
    """ The messages pylint prints on a file, under no configuration file. """
    empty = path.parent / "pylintrc"
    empty.write_text("")
    return subprocess.run(
        [sys.executable, "-m", "pylint", f"--rcfile={empty}", "--score=no",
         "--disable=all", "--enable=no-self-argument,no-member",
         *options, str(path)],
        capture_output=True, text=True, check=False).stdout


def test_axiom_is_a_classmethod(tmp_path):
    path = tmp_path / "law.py"
    path.write_text(LAW)
    assert "no-self-argument" in messages(path)
    assert not messages(
        path, f"--init-hook=import sys; sys.path.append({str(SCRIPTS)!r})",
        "--load-plugins=pylint_axioms").strip()
