"""Tests for the transcript of everything said on a pull request."""

import pytest


def comment(when, body, login="toumix", **rest):
    return dict({"created_at": when, "body": body, "id": hash(body) % 1000,
                 "user": {"login": login}}, **rest)


@pytest.fixture
def discussion():
    """A conversation comment, one review thread of a remark and the
    answer it drew, and a submitted review."""
    remark = comment("2026-08-27T09:00:00Z", "a remark", path="a.py", line=3)
    return (
        [comment("2026-08-27T10:00:00Z", "a question")],
        [remark, comment("2026-08-27T11:00:00Z", "an answer", path="a.py",
                         line=3, in_reply_to_id=remark["id"])],
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
    assert "earlier message" in transcript
    assert "a remark" not in transcript
    assert "an answer" in transcript


def test_render_protects_the_last_word_on_an_open_flag(thread, discussion):
    """The reply a remark drew is what stops it being raised again, so it
    is the last thing dropped rather than the first."""
    transcript = thread.render(thread.entries(*discussion), budget=150)
    assert "an answer" in transcript
    assert "a remark" not in transcript


def test_each_thread_keeps_its_own_last_word(thread):
    """Two threads can sit on one line; sparing one of them is not
    sparing the other."""
    first = comment("2026-08-27T09:00:00Z", "a remark", path="a.py", line=3)
    second = comment("2026-08-27T09:30:00Z", "another remark",
                     path="a.py", line=3)
    answers = [
        comment("2026-08-27T10:00:00Z", "answering the first",
                path="a.py", line=3, in_reply_to_id=first["id"]),
        comment("2026-08-27T10:30:00Z", "answering the second",
                path="a.py", line=3, in_reply_to_id=second["id"])]
    items = thread.entries([], [first, second] + answers, [])
    transcript = thread.render(items, budget=200)
    assert "answering the first" in transcript
    assert "answering the second" in transcript
    assert "a remark" not in transcript


def test_render_never_hands_back_more_than_the_budget(thread, discussion):
    """What it spares is spared while anything else is left, not past the
    budget: an oversized transcript is dropped whole by the prompt."""
    items = thread.entries(*discussion)
    for budget in (10, 50, 120, 250):
        assert len(thread.render(items, budget)) <= budget


def test_render_counts_the_note_it_adds(thread, discussion):
    items = thread.entries(*discussion)
    assert len(thread.render(items, budget=300)) <= 300
