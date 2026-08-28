"""Tests for what the style reviewer remembers of its past rounds."""


def review(history, remarks, id, body="Style review by `m`."):
    return {"id": id, "body": history.stamp(remarks) + "\n" + body,
            "submitted_at": f"2026-08-27T{id:02d}:00:00Z",
            "state": "COMMENTED", "user": {"type": "Bot", "login": "discopy"}}


def test_stamp_survives_a_remark_quoting_it(history):
    """The record is an HTML comment, so no `>` may reach it unescaped."""
    remarks = [{"path": "a.py", "line": 3, "comment": "write --> not ->"}]
    stamped = history.stamp(remarks)
    assert ">" not in stamped[:-len(" -->")]
    assert history.recorded(stamped + "\nStyle review.") == remarks


def test_recorded_reads_somebody_else_s_review_as_none(history):
    assert history.recorded("LGTM") is None
    assert history.recorded("") is None


def test_recorded_reads_an_unparsable_marker_as_none(history):
    """One malformed body must not cost every later round its history."""
    assert history.recorded(history.MARKER + "{not json -->") is None
    assert history.recorded(history.MARKER + "no closing bracket") is None


def test_history_numbers_the_remarks_across_the_rounds(history, monkeypatch):
    first = [{"path": "a.py", "line": 3, "comment": "one"}]
    second = [{"path": "b.py", "line": 9, "comment": "two"},
              {"path": "b.py", "line": 12, "comment": "three"}]
    listed = {
        "/repos/o/r/pulls/1/reviews": [
            {"id": 1, "body": "not ours", "state": "APPROVED",
             "submitted_at": "2026-08-27T00:00Z",
             "user": {"type": "User", "login": "toumix"}},
            review(history, first, 2), review(history, second, 3)],
        "/repos/o/r/pulls/1/comments": [],
        "/repos/o/r/issues/1/comments": []}
    monkeypatch.setattr(history, "listing", lambda path, token: listed[path])
    past = history.history("o/r", "1", "token")
    assert past["rounds"] == 2
    assert [remark["number"] for remark in past["remarks"]] == [1, 2, 3]
    assert [remark["comment"] for remark in past["remarks"]] == [
        "one", "two", "three"]
    assert [carrier["id"] for carrier in past["reviews"]] == [2, 3]


def test_history_of_a_pull_request_nobody_reviewed(history, monkeypatch):
    monkeypatch.setattr(history, "listing", lambda path, token: [])
    assert history.history("o/r", "1", "token") == history.empty()
