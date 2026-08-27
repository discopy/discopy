"""Tests for the benchmark comparison comment.

A ``workflow_run`` job runs with the privileges of the default branch on an
artifact staged by a run that has fewer, so most of these state what the
job refuses to post.
"""

import pytest

BASE = "a" * 40
PREVIOUS = "b" * 40
HEAD = "c" * 40


@pytest.fixture
def data():
    return {"pull_request": 7, "base": BASE, "previous": PREVIOUS,
            "head": HEAD, "run_id": 99}


@pytest.fixture
def run():
    return {
        "id": 99, "head_sha": HEAD, "head_branch": "a-branch",
        "conclusion": "success", "html_url": "https://run",
        "head_repository": {"full_name": "discopy/discopy"},
        "pull_requests": [
            {"number": 7, "head": {"sha": HEAD}, "base": {"sha": BASE}}]}


@pytest.fixture
def pull():
    repo = {"full_name": "discopy/discopy", "html_url": "https://repo"}
    return {"head": {"sha": HEAD, "ref": "a-branch", "repo": repo},
            "base": {"sha": BASE, "repo": repo}}


def test_the_honest_metadata_passes(benchmark_comment, data, run, pull):
    assert benchmark_comment.unreadable(data, run) is None
    assert benchmark_comment.mismatch(
        data, run, pull, "discopy/discopy") is None


@pytest.mark.parametrize("field,value", [
    ("pull_request", "7"), ("pull_request", True), ("pull_request", None),
    ("run_id", 100), ("base", "z" * 40), ("previous", ""), ("head", "c" * 39),
    ("head", "d" * 40)])
def test_unreadable_rejects_a_forged_field(
        benchmark_comment, data, run, field, value):
    assert benchmark_comment.unreadable(
        dict(data, **{field: value}), run) == "Invalid benchmark metadata."


def test_mismatch_rejects_a_foreign_head_repository(
        benchmark_comment, data, run, pull):
    pull["head"]["repo"] = {"full_name": "someone/fork", "html_url": "x"}
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark metadata does not match its source PR.")


def test_mismatch_rejects_another_branch(
        benchmark_comment, data, run, pull):
    pull["head"]["ref"] = "another-branch"
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark metadata does not match its source PR.")


def test_mismatch_rejects_a_run_belonging_elsewhere(
        benchmark_comment, data, run, pull):
    run["pull_requests"] = [
        {"number": 8, "head": {"sha": HEAD}, "base": {"sha": BASE}}]
    assert benchmark_comment.mismatch(data, run, pull, "discopy/discopy") == (
        "Benchmark run does not belong to this PR.")


def test_mismatch_accepts_a_run_listing_no_pull_request(
        benchmark_comment, data, run, pull):
    """A run from a fork lists none, and the checks above already tie it
    to this pull request."""
    run["pull_requests"] = []
    assert benchmark_comment.mismatch(
        data, run, pull, "discopy/discopy") is None


def test_sanitised_neutralises_html_and_mentions(benchmark_comment):
    assert benchmark_comment.sanitised("<b> @toumix") == "&lt;b> &#64;toumix"


def test_body_links_the_merge_base(benchmark_comment, data, run, pull):
    body = benchmark_comment.body(data, run, pull, "a report")
    assert body.startswith(benchmark_comment.MARKER)
    assert "a report" in body
    assert f"merge base [`{PREVIOUS[:7]}`](https://repo/commit/{PREVIOUS})" \
        in body
    assert "[!WARNING]" not in body


def test_body_warns_when_the_base_moved(benchmark_comment, data, run, pull):
    pull["base"]["sha"] = "e" * 40
    run["conclusion"] = "failure"
    body = benchmark_comment.body(data, run, pull, "a report")
    assert "[!WARNING]" in body
    assert "base has changed" in body
    assert "concluded `failure`" in body


def test_ours_is_the_marked_comment(benchmark_comment):
    marker = benchmark_comment.MARKER
    comments = [
        {"id": 1, "user": {"login": "toumix"}, "body": f"{marker} a forgery"},
        {"id": 2, "user": {"login": "github-actions[bot]"}, "body": "hello"},
        {"id": 3, "user": {"login": "github-actions[bot]"},
         "body": f"{marker}\nthe comparison"}]
    assert benchmark_comment.ours(comments)["id"] == 3
    assert benchmark_comment.ours(comments[:2]) is None
