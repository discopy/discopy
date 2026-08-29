"""Tests for the style reviewer's assembly of its one request."""

import subprocess

import pytest


def git(*args, cwd):
    subprocess.run(("git", ) + args, cwd=cwd, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repository(tmp_path):
    """A repository with one commit on a base branch and one on top."""
    git("init", "-q", "-b", "main", ".", cwd=tmp_path)
    git("config", "user.email", "a@b.c", cwd=tmp_path)
    git("config", "user.name", "Tester", cwd=tmp_path)
    (tmp_path / "file.py").write_text("kept\nremoved\nkept\n")
    git("add", ".", cwd=tmp_path)
    git("commit", "-qm", "base", cwd=tmp_path)
    base = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=tmp_path, check=True,
        capture_output=True, text=True).stdout.strip()
    (tmp_path / "file.py").write_text("kept\nadded\nkept\n")
    git("commit", "-qam", "head", cwd=tmp_path)
    return tmp_path, base


def test_annotated_numbers_the_new_file(review, repository, monkeypatch):
    """A removed line carries no number, since it has none in the new
    file; only an added line gets a leading ``+``."""
    path, base = repository
    monkeypatch.chdir(path)
    assert review.annotated("file.py", base) == "\n".join([
        "1  kept", "  -removed", "2 +added", "3  kept"])


def test_fence_outlasts_the_longest_run(review):
    assert review.fence("no ticks") == "```"
    assert review.fence("a ```python cell``` inside") == "````"
    assert review.fence("````four````") == "`````"


def test_section_is_fenced_by_the_file_type(review):
    body = "```python {.marimo}\nx = 1\n```\n"
    section = review.section("Changed", "docs/notebooks/a.md", body)
    assert section.startswith("# Changed: docs/notebooks/a.md\n\n````markdown")
    assert section.endswith("````")


def test_language_falls_back_to_text(review):
    assert review.language("a.py") == "python"
    assert review.language("a.yml") == "yaml"
    assert review.language("uv.lock") == "text"


def test_contents_budgets_the_separator_too(review):
    """The ``"\\n\\n"`` that joins a block to its neighbour is part of what
    the block costs: #611 was every part being budgeted raw."""
    kept, left, dropped = review.contents(
        ["a", "b"], budget=12, block=lambda path: path * 5)
    assert [path for path, _ in kept] == ["a"]
    assert (left, dropped) == (5, ["b"])


def test_imports_finds_the_package_local_files(review, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "there.py").write_text("")
    (tmp_path / "pack" / "here.py").write_text(
        "import os\nfrom . import there\nfrom pack.there import thing\n")
    assert review.imports("pack/here.py") == ["pack/there.py"]


def test_imports_gives_up_on_what_python_cannot_parse(review, tmp_path):
    notebook = tmp_path / "a.md"
    notebook.write_text("# A notebook\n\n```python {.marimo}\nx = 1\n```\n")
    assert review.imports(str(notebook)) == []


def test_imports_gives_up_on_binary_files(review, tmp_path):
    cache = tmp_path / "a.npz"
    cache.write_bytes(b"PK\x03\x04" + bytes(range(256)))
    assert review.imports(str(cache)) == []
