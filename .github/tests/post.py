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


def test_commentable_lines(post):
    assert post.commentable_lines(DIFF) == {
        "discopy/cat.py": {12, 13, 14, 15, 50, 51}}


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


def test_describe_counts_what_it_drops(post, monkeypatch):
    monkeypatch.setenv("MODEL", "a-model")
    body = post.describe(
        [{"path": "a.py", "line": 1, "comment": "x"}], withheld=3,
        unreadable=2)
    assert "`a.py:1` — x" in body
    assert "3 more past the ten-finding cap" in body
    assert "2 unreadable findings dropped" in body
