"""Tests for the pylint plugin reading an ``@axiom`` law as a classmethod."""

import pathlib
import subprocess
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"

LAW = '''
def axiom(equation):
    return equation


class Meta(type):
    attribute = 1


class Category(metaclass=Meta):
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
    """
    Without the plugin, the law is a method wanting ``self`` and its first
    argument an instance, which has no member of the metaclass; with it,
    the argument is the class and the member is found.
    """
    path = tmp_path / "law.py"
    path.write_text(LAW)
    plain = messages(path)
    assert "no-self-argument" in plain and "no-member" in plain
    assert not messages(
        path, f"--init-hook=import sys; sys.path.append({str(SCRIPTS)!r})",
        "--load-plugins=pylint_axioms").strip()
