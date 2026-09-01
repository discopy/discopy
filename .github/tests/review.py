"""Tests for the style reviewer's assembly of its one request."""

import http.client
import io
import json
import os
import subprocess
import urllib.error

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


def test_past_block_numbers_the_remarks_as_the_verdicts_do(review):
    block = review.past_block([
        {"number": 1, "path": "a.py", "line": 3, "comment": "one"},
        {"number": 2, "path": "b.py", "line": 9, "comment": "two"}])
    assert "1. `a.py:3` — 'one'" in block
    assert "2. `b.py:9` — 'two'" in block


def test_a_remark_reaches_the_prompt_as_one_line(review):
    """A newline in a remark would break the numbered listing it sits
    in, so it goes in as a literal."""
    block = review.past_block(
        [{"number": 1, "path": "a.py", "line": 3, "comment": "one\ntwo"}])
    assert block.count("\n") == 2
    assert "'one\\ntwo'" in block


def test_literal_bounds_what_somebody_wrote(review):
    assert len(review.literal("x" * 5_000)) < review.QUOTE + 10


def test_fitted_drops_what_does_not_fit_whole(review):
    assert review.fitted("abc", 5) == ("abc", 0)
    assert review.fitted("abc", 4) == ("", 4)


@pytest.fixture
def reviewable(repository, monkeypatch):
    """A repository the reviewer can assemble a prompt in."""
    tmp_path, base = repository
    (tmp_path / ".github" / "style-review").mkdir(parents=True)
    (tmp_path / ".github" / "style-review" / "prompt.md").write_text(
        "instructions\n")
    (tmp_path / "STYLE.md").write_text("the style guide\n")
    monkeypatch.chdir(tmp_path)
    return base


def past(remarks, discussion=""):
    return {"remarks": remarks, "discussion": discussion}


def test_assemble_orders_from_what_never_moves_to_what_always_does(
        review, reviewable):
    """The prompt is a prefix two rounds share, so the parts that move
    every round go last."""
    prompt, _ = review.assemble(["file.py"], reviewable, past(
        [{"number": 1, "path": "file.py", "line": 2, "comment": "one"}],
        "### toumix\n\nno, on purpose"))
    places = [prompt.index(part) for part in (
        "# STYLE.md", "# Style remarks from the previous rounds",
        "# Discussion so far", "# Changed: file.py")]
    assert places == sorted(places)


def test_assemble_adds_to_the_round_before_it(review, reviewable):
    """What two rounds share is a prefix reaching the whole of the
    earlier round's remarks, so a gateway can serve it from its cache."""
    first, _ = review.assemble(["file.py"], reviewable, past(
        [{"number": 1, "path": "file.py", "line": 2, "comment": "one"}]))
    second, _ = review.assemble(["file.py"], reviewable, past([
        {"number": 1, "path": "file.py", "line": 2, "comment": "one"},
        {"number": 2, "path": "file.py", "line": 3, "comment": "two"}]))
    shared = os.path.commonprefix([first, second])
    assert "1. `file.py:2` — 'one'" in shared
    assert "# Changed: file.py" not in shared


def test_a_size_note_lands_with_the_files_it_names(
        review, reviewable, monkeypatch):
    """The notes say what did not fit this round, so they go after the
    prefix two rounds share rather than in its middle."""
    monkeypatch.setattr(review, "BUDGET", 1_000)
    with open("file.py", "a") as file:
        file.write("filler\n" * 500)
    prompt, _ = review.assemble(["file.py"], reviewable, past(
        [{"number": 1, "path": "file.py", "line": 2, "comment": "one"}],
        "### toumix\n\nno, on purpose"))
    places = [prompt.index(part) for part in (
        "# Style remarks from the previous rounds", "# Discussion so far",
        "# Changed files past the budget even as a diff")]
    assert places == sorted(places)


class Cut:
    """A gateway that cuts the transfer short before answering."""

    def __init__(self, answer, failures):
        self.answer, self.failures, self.attempts = answer, failures, 0

    def __call__(self, request, timeout=None):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise http.client.IncompleteRead(b"half an ans")
        return io.BytesIO(json.dumps(self.answer).encode())


def test_complete_reads_again_when_the_transfer_is_cut_short(
        review, monkeypatch):
    gateway = Cut({"choices": []}, failures=1)
    monkeypatch.setattr(review.urllib.request, "urlopen", gateway)
    assert review.complete("request") == {"choices": []}
    assert gateway.attempts == 2


def test_complete_gives_up_after_its_attempts(review, monkeypatch):
    gateway = Cut({}, failures=review.ATTEMPTS)
    monkeypatch.setattr(review.urllib.request, "urlopen", gateway)
    with pytest.raises(http.client.IncompleteRead):
        review.complete("request")
    assert gateway.attempts == review.ATTEMPTS


def test_complete_asks_once_when_the_gateway_answers_an_error(
        review, monkeypatch):
    """An `HTTPError` is the gateway answering, not the transfer
    failing, and it is a subclass of the `URLError` retried above."""
    asked = []

    def refuse(request, timeout=None):
        asked.append(request)
        raise urllib.error.HTTPError("url", 400, "Bad Request", {}, None)

    monkeypatch.setattr(review.urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.HTTPError):
        review.complete("request")
    assert len(asked) == 1
