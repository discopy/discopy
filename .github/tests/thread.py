"""Tests for the transcript of everything said on a pull request."""

import pytest


def comment(when, body, login="toumix", **rest):
    return dict({"created_at": when, "body": body,
                 "user": {"login": login}}, **rest)


@pytest.fixture
def discussion():
    return (
        [comment("2026-08-27T10:00:00Z", "a question")],
        [comment("2026-08-27T09:00:00Z", "a remark", path="a.py", line=3),
         comment("2026-08-27T11:00:00Z", "an answer", path="a.py", line=3)],
        [{"submitted_at": "2026-08-27T12:00:00Z", "state": "APPROVED",
          "body": "looks good", "user": {"login": "toumix"}}])


def test_entries_are_ordered_oldest_first(thread, discussion):
    assert [entry["when"] for entry in thread.entries(*discussion)] == [
        "2026-08-27T09:00:00Z", "2026-08-27T10:00:00Z",
        "2026-08-27T11:00:00Z", "2026-08-27T12:00:00Z"]


def test_a_diff_comment_carries_where_it_was_made(thread, discussion):
    anchors = [entry["anchor"] for entry in thread.entries(*discussion)]
    assert anchors == ["a.py:3", None, "a.py:3", None]


def test_anchor_falls_back_to_the_line_a_comment_was_made_on(thread):
    assert thread.anchor(
        {"path": "a.py", "line": None, "original_line": 7}) == "a.py:7"
    assert thread.anchor({"path": "a.py", "line": None}) == "a.py"


def test_a_pending_review_is_not_a_contribution(thread):
    assert thread.entries([], [], [
        {"submitted_at": None, "state": "PENDING", "body": "drafting",
         "user": {"login": "toumix"}}]) == []


def test_render_drops_the_oldest_first_and_says_how_many(thread, discussion):
    items = thread.entries(*discussion)
    transcript = thread.render(items, budget=200)
    assert "1 earlier message omitted for size" in transcript
    assert "a remark" not in transcript
    assert "a question" in transcript


def test_render_protects_the_last_word_on_an_open_flag(thread, discussion):
    """The reply a remark drew is what stops it being raised again."""
    transcript = thread.render(thread.entries(*discussion), budget=10)
    assert "an answer" in transcript
