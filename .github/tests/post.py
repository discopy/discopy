"""Tests for the style reviewer's posting of its findings."""

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


def test_summary_counts_every_remark(post):
    assert post.summary(["accepted", "declined"]) == (
        "2 style remarks: 1 accepted / 1 declined")
    assert post.summary(["accepted", None]) == (
        "2 style remarks: 1 accepted / 0 declined / 1 still open")
    assert post.summary(["declined"]) == (
        "1 style remark: 0 accepted / 1 declined")
    assert post.summary([]) is None


def test_verdicts_read_what_this_round_answered(post):
    past = {"remarks": [{"number": 1}, {"number": 2}], "verdicts": {}}
    kept = post.verdicts(past, [{"remark": 2, "verdict": "declined"}])
    assert post.tally(past["remarks"], kept) == [None, "declined"]
    kept = post.verdicts(past, [{"remark": "1", "verdict": "accepted"},
                                {"bad": "shape"}, None])
    assert post.tally(past["remarks"], kept) == ["accepted", None]


def test_tallied_replaces_the_tally_it_finds(post, history):
    body = history.stamp([]) + "\nStyle review by `m`, round 1."
    once = post.tallied(body, "1 style remark: 1 accepted / 0 declined")
    assert once.endswith("1 style remark: 1 accepted / 0 declined")
    twice = post.tallied(once, "2 style remarks: 2 accepted / 0 declined")
    assert twice == post.tallied(body, "2 style remarks: 2 accepted / 0 "
                                 "declined")
    assert post.tallied(twice, None) == body


def test_tallied_leaves_a_remark_quoting_the_marker_alone(post, history):
    """The tally is stripped from the foot, not from wherever the marker
    is mentioned: a remark about the tally is a remark like any other."""
    body = (history.stamp([]) + "\nStyle review.\nnever write "
            + history.TALLY + " in a review")
    assert post.tallied(post.tallied(body, "1 style remark: 1 accepted / 0 "
                                     "declined"), None) == body


def test_a_verdict_survives_a_round_that_forgets_it(post):
    """A remark somebody accepted does not go back to open because a
    later round could no longer see the file it was about."""
    past = {"remarks": [{"number": 1}, {"number": 2}],
            "verdicts": {"1": "accepted", "2": "declined"}}
    kept = post.verdicts(past, [])
    assert post.tally(past["remarks"], kept) == ["accepted", "declined"]
    kept = post.verdicts(past, [{"remark": 1, "verdict": "open"}])
    assert post.tally(past["remarks"], kept) == ["accepted", "declined"]


def test_a_decisive_verdict_overrides_the_one_before_it(post):
    past = {"remarks": [{"number": 1}], "verdicts": {"1": "declined"}}
    kept = post.verdicts(past, [{"remark": 1, "verdict": "accepted"}])
    assert post.tally(past["remarks"], kept) == ["accepted"]


def test_a_remark_nobody_has_answered_is_open(post):
    past = {"remarks": [{"number": 1}, {"number": 2}],
            "verdicts": {"1": "accepted"}}
    kept = post.verdicts(past, [{"bad": "shape"}, None])
    assert post.tally(past["remarks"], kept) == ["accepted", None]


def test_the_tally_carries_its_verdicts_for_the_next_round(post, history):
    body = history.stamp([]) + "\nStyle review by `m`, round 1."
    tallied = post.tallied(body, "1 style remark: 1 accepted / 0 declined",
                           {"1": "accepted"})
    assert history.scored(tallied) == {"1": "accepted"}
    assert post.tallied(tallied, None) == body
