"""Tests for the style reviewer's posting of its findings."""

import json
import os

import pytest

DIFF = """\
diff --git a/discopy/cat.py b/discopy/cat.py
--- a/discopy/cat.py
+++ b/discopy/cat.py
@@ -10,3 +12,4 @@ class Ob:
 context
+added
 context
 context
@@ -40 +50,2 @@
+added
 context
"""


def test_commentable_lines_are_every_line_a_hunk_shows(post):
    """The prompt asks for findings on what the diff adds and allows the
    surrounding lines as an exception, so what can be said inline is
    every line GitHub takes a comment on."""
    assert post.commentable_lines(DIFF) == {
        "discopy/cat.py": {12, 13, 14, 15, 50, 51}}


def test_commentable_lines_ignores_a_hunk_before_any_file(post):
    assert post.commentable_lines("@@ -1 +1 @@\n context\n") == {}


TWO_FILES = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
 context
+added
diff --git a/b.py b/b.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/b.py
@@ -0,0 +1,2 @@
+first
+second
"""


def test_commentable_lines_does_not_leak_across_a_file_boundary(post):
    """Inter-file metadata rows (`diff --git`, `index`, `new file mode`)
    used to fall through to the previous file's counter, inflating its
    line numbers past its own end."""
    assert post.commentable_lines(TWO_FILES) == {
        "a.py": {1, 2}, "b.py": {1, 2}}


def test_normalised_reads_a_string_line(post):
    finding = {"path": "discopy/cat.py", "line": "42", "comment": " x "}
    assert post.normalised(finding) == {
        "path": "discopy/cat.py", "line": 42, "comment": "x"}


def test_normalised_rejects_the_unreadable(post):
    for finding in [
            {"path": "a.py", "line": True, "comment": "x"},
            {"path": "a.py", "line": None, "comment": "x"},
            {"path": "a.py", "line": "middle", "comment": "x"},
            {"path": "a.py", "line": 1, "comment": "   "},
            {"path": 42, "line": 1, "comment": "x"},
            {"line": 1, "comment": "x"},
            {"path": "a.py", "line": 1}]:
        assert post.normalised(finding) is None


def test_describe_counts_what_it_could_not_say(post, monkeypatch):
    monkeypatch.setenv("MODEL", "a-model")
    body = post.describe(3, withheld=3, unreadable=1, coverage={
        "degraded": ["a.py"], "unreviewed": ["b.py", "c.py"]})
    assert "round 3." in body
    assert "3 further findings went past the ten-finding cap." in body
    assert "1 finding could not be read." in body
    assert "reviewed from their diff alone: `a.py`." in body
    assert "not reviewed at all: `b.py`, `c.py`." in body


def test_describe_lists_no_finding(post, monkeypatch):
    """A remark goes on its line wherever GitHub takes it there, so the
    body lists only what it has to carry."""
    monkeypatch.setenv("MODEL", "a-model")
    assert post.describe(1, withheld=0, unreadable=0, coverage={}) == (
        "Style review by `a-model`, round 1.")


def test_elsewhere_names_where_each_remark_is_about(post):
    assert post.elsewhere([{"path": "a.py", "line": 3, "comment": "one"}],
                          "why:") == ["", "why:", "- `a.py:3` — one"]


def test_counted_says_one_of_a_thing_singular(post):
    assert post.counted(1, "finding") == "1 finding"
    assert post.counted(0, "finding") == "0 findings"
    assert post.counted(2, "style remark") == "2 style remarks"


def scored(post, given):
    """The tally line of a round that made as many remarks as there are
    verdicts, numbered from one."""
    return post.summary(range(1, len(given) + 1), {
        str(number): verdict for number, verdict in enumerate(given, 1)
        if verdict is not None})


def test_summary_counts_every_remark(post):
    assert scored(post, ["accepted", "declined"]) == (
        "2 style remarks: 1 accepted / 1 declined")
    assert scored(post, ["accepted", "declined", None]) == (
        "3 style remarks: 1 accepted / 1 declined / 1 still open")
    assert scored(post, []) is None


def test_summary_leaves_out_a_state_nothing_is_in(post):
    """Nought declined is not news; what every remark is, is."""
    assert scored(post, ["accepted", None]) == (
        "2 style remarks: 1 accepted / 1 still open")
    assert scored(post, ["accepted"] * 3) == "3 style remarks: all accepted"
    assert scored(post, [None, None]) == "2 style remarks: all still open"


def test_summary_says_of_one_remark_what_became_of_it(post):
    assert scored(post, ["accepted"]) == "1 style remark: accepted"
    assert scored(post, ["declined"]) == "1 style remark: declined"


def test_a_round_counts_its_own_remarks_and_no_others(post):
    """Two rounds of one remark each are two tallies of one, not one of
    two: a review says how what it asked for landed."""
    verdicts = {"1": "accepted", "2": "declined"}
    assert post.summary([1], verdicts) == "1 style remark: accepted"
    assert post.summary([2], verdicts) == "1 style remark: declined"
    assert post.answered([1], verdicts) == {"1": "accepted"}
    assert post.answered([2, 3], verdicts) == {"2": "declined"}


def test_verdicts_read_what_this_round_answered(post):
    past = {"verdicts": {}}
    kept = post.verdicts(past, [{"remark": 2, "verdict": "declined"}])
    assert post.summary([1, 2], kept) == (
        "2 style remarks: 1 declined / 1 still open")
    kept = post.verdicts(past, [{"remark": "1", "verdict": "accepted"},
                                {"bad": "shape"}, None])
    assert post.summary([1, 2], kept) == (
        "2 style remarks: 1 accepted / 1 still open")


def test_tallied_replaces_the_tally_it_finds(post, history):
    body = history.stamp([]) + "\nStyle review by `m`, round 1."
    once = post.tallied(body, "1 style remark: accepted")
    assert once.endswith("1 style remark: accepted")
    twice = post.tallied(once, "2 style remarks: all accepted")
    assert twice == post.tallied(body, "2 style remarks: all accepted")
    assert post.tallied(twice, None) == body


def test_tallied_leaves_a_remark_quoting_the_marker_alone(post, history):
    """The tally is stripped from the foot, not from wherever the marker
    is mentioned: a remark about the tally is a remark like any other."""
    body = (history.stamp([]) + "\nStyle review.\nnever write "
            + history.TALLY + " in a review")
    assert post.tallied(
        post.tallied(body, "1 style remark: accepted"), None) == body


def test_tallied_survives_a_remark_quoting_it_after_a_blank_line(
        post, history):
    """A remark quoting the marker after a blank line used to be read as
    the real tally, and everything from there to the end of the body
    silently discarded — real review content, not just the marker."""
    body = (history.stamp([]) + "\nStyle review.\n\nnever write "
            + history.TALLY + " after a blank line"
            + "\n\nreal content that must survive")
    once = post.tallied(body, "1 style remark: accepted")
    assert once.startswith(body)
    assert once.endswith("1 style remark: accepted")


def test_a_verdict_survives_a_round_that_forgets_it(post):
    """A remark somebody accepted does not go back to open because a
    later round could no longer see the file it was about."""
    past = {"verdicts": {"1": "accepted", "2": "declined"}}
    assert post.verdicts(past, []) == past["verdicts"]
    assert post.verdicts(
        past, [{"remark": 1, "verdict": "open"}]) == past["verdicts"]


def test_a_decisive_verdict_overrides_the_one_before_it(post):
    past = {"verdicts": {"1": "declined"}}
    kept = post.verdicts(past, [{"remark": 1, "verdict": "accepted"}])
    assert kept == {"1": "accepted"}


def test_a_remark_nobody_has_answered_is_open(post):
    past = {"verdicts": {"1": "accepted"}}
    kept = post.verdicts(past, [{"bad": "shape"}, None])
    assert post.summary([1, 2], kept) == (
        "2 style remarks: 1 accepted / 1 still open")


def test_the_tally_carries_its_verdicts_for_the_next_round(post, history):
    body = history.stamp([]) + "\nStyle review by `m`, round 1."
    tallied = post.tallied(body, "1 style remark: accepted",
                           {"1": "accepted"})
    assert history.scored(tallied) == {"1": "accepted"}
    assert post.tallied(tallied, None) == body


def test_main_tallies_each_round_onto_the_round_that_made_it(
        post, history, tmp_path, monkeypatch):
    """Two rounds of one remark each get one tally each, and the round
    being posted gets none: it is scored by the ones that follow it."""
    monkeypatch.chdir(tmp_path)
    os.makedirs(history.DIRECTORY)
    rounds = [{"id": 1, "body": "round 1", "numbers": [1]},
              {"id": 2, "body": "round 2", "numbers": [2]}]
    with open(os.path.join(history.DIRECTORY, "history.json"), "w") as file:
        json.dump({"rounds": rounds, "remarks": [], "discussion": "",
                   "verdicts": {"1": "accepted"}}, file)
    with open(os.path.join(history.DIRECTORY, "findings.json"), "w") as file:
        json.dump({"findings": [], "verdicts": [
            {"remark": 2, "verdict": "declined"}]}, file)
    with open(os.path.join(history.DIRECTORY, "diff.patch"), "w") as file:
        file.write(DIFF)
    edited = []
    monkeypatch.setattr(post, "moved", lambda: None)
    monkeypatch.setattr(post, "rewrite", lambda review, body: edited.append(
        (review["id"], body)))
    post.main()
    assert [number for number, _ in edited] == [1, 2]
    assert edited[0][1].endswith("1 style remark: accepted")
    assert edited[1][1].endswith("1 style remark: declined")
    assert history.scored(edited[0][1]) == {"1": "accepted"}
    assert history.scored(edited[1][1]) == {"2": "declined"}


def test_main_stands_down_on_a_moved_head_and_is_not_clean(
        post, tmp_path, monkeypatch):
    """A round that reviewed nothing is not a round that found nothing:
    the workflow calls the correctness reviewer once per pull request,
    and an unset output is read as clean, so it would spend that call on
    a head nobody read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(post, "moved", lambda: "f" * 40)
    monkeypatch.setattr(post, "rewrite", lambda review, body: pytest.fail(
        "a round that stood down edits nothing"))
    post.main()
    assert open(tmp_path / "output").read() == "clean=false\n"


def staged(history, path, findings, coverage=None):
    """A round's inputs on disk, as the workflow's steps hand them over."""
    os.makedirs(history.DIRECTORY, exist_ok=True)
    with open(os.path.join(history.DIRECTORY, "history.json"), "w") as file:
        json.dump(history.empty(), file)
    with open(os.path.join(history.DIRECTORY, "findings.json"), "w") as file:
        json.dump({"findings": findings, "coverage": coverage or {}}, file)
    with open(os.path.join(history.DIRECTORY, "diff.patch"), "w") as file:
        file.write(DIFF)


def test_main_posts_a_finding_off_the_diff_in_the_body(
        post, history, tmp_path, monkeypatch):
    """Commenting outside the diff is discouraged, not forbidden: the
    remark is inline where GitHub takes it there and in the body
    otherwise, rather than dropped."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL", "a-model")
    staged(history, tmp_path, [
        {"path": "discopy/cat.py", "line": 13, "comment": "on the diff"},
        {"path": "discopy/cat.py", "line": 12, "comment": "around it"},
        {"path": "discopy/cat.py", "line": 900, "comment": "far away"}])
    posted = []
    monkeypatch.setattr(post, "moved", lambda: None)
    monkeypatch.setattr(post, "post_review", lambda body, remarks, comments:
                        posted.append((body, remarks, comments)))
    post.main()
    body, remarks, comments = posted[0]
    assert [f["line"] for f in comments] == [13, 12]
    assert [f["line"] for f in remarks] == [13, 12, 900]
    assert "- `discopy/cat.py:900` — far away" in body
    assert "on the diff" not in body


def test_main_says_what_it_could_not_read_with_nothing_to_report(
        post, history, tmp_path, monkeypatch):
    """A round that read no finding but could not read a file whole is
    not a clean one: it hands over to the correctness reviewer as clean
    otherwise, on a partial read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL", "a-model")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    staged(history, tmp_path, [], {"unreviewed": ["huge.py"]})
    posted = []
    monkeypatch.setattr(post, "moved", lambda: None)
    monkeypatch.setattr(post, "post_review", lambda body, remarks, comments:
                        posted.append((body, remarks, comments)))
    post.main()
    body, remarks, comments = posted[0]
    assert (remarks, comments) == ([], [])
    assert "not reviewed at all: `huge.py`." in body
    assert open(tmp_path / "output").read() == "clean=false\n"
