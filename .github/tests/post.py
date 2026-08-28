"""Tests for the style reviewer's posting of its findings."""

import json
import os

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


def test_commentable_lines_are_the_lines_the_diff_adds(post):
    """A hunk shows its surroundings and GitHub would take a comment on
    them, but the prompt asks for a finding on a line the diff adds."""
    assert post.commentable_lines(DIFF) == {"discopy/cat.py": {13, 50}}


def test_commentable_lines_ignores_a_hunk_before_any_file(post):
    assert post.commentable_lines("@@ -1 +1 @@\n context\n") == {}


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
    body = post.describe(3, dropped=2, withheld=3, unreadable=1)
    assert "round 3." in body
    assert "2 findings sat on no line of the diff." in body
    assert "3 further findings went past the ten-finding cap." in body
    assert "1 finding could not be read." in body


def test_describe_lists_no_finding(post, monkeypatch):
    """Every remark is an inline comment, so the body never lists one."""
    monkeypatch.setenv("MODEL", "a-model")
    assert post.describe(1, dropped=0, withheld=0, unreadable=0) == (
        "Style review by `a-model`, round 1.")


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
