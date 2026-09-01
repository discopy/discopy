"""Post the style reviewer's findings as one pull request review.

Reads ``.style-review/findings.json`` and ``.style-review/diff.patch``,
posts the findings sitting on a line of the diff as inline comments and the
others in the review body, as the GitHub App authenticated by ``APP_TOKEN``.
A clean run posts nothing.

A remark belongs on the line it is about, and the review asks for the
diff ([#673](https://github.com/discopy/discopy/issues/673)) — but
commenting outside it is an exception the reviewer may take, not a
mistake: GitHub takes a comment on any line one of the diff's hunks
shows, so a remark on the code the change is read against goes inline
like the rest, and one further out still goes in the body rather than
be lost. The body is where the round says what it could not say inline:
the remarks GitHub would not take, the findings past the cap, and the
changed files too big to be read whole.

One review per round, and the round records the remarks it made so that the
next one can read them back. Whatever the model makes of those past remarks
— accepted, declined or still open — is tallied onto the round that made
them, each round counting its own remarks and no others, so that a review
says how what it asked for landed rather than how every round's did. A
round is scored by the ones that follow it, so the review being posted
carries no tally of its own yet.

A round whose head has moved posts nothing: its findings are about lines
somebody has already replaced, and the push that replaced them starts a
round of its own. It reports itself as not clean, since a round that
reviewed nothing is not one that found nothing: the workflow calls the
correctness reviewer once per pull request, and calling it here would
spend that on a head nobody read.
"""

import json
import os
import re
import urllib.error

import history
from github import api

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def commentable_lines(diff):
    """Map each path in a unified diff to the lines GitHub takes a
    comment on: every line one of its hunks shows, the ones the change
    adds and the ones it is read against alike. The prompt asks for
    findings on what the diff adds and allows the rest as an exception,
    so this is the wider set — what can be said inline at all. A
    ``diff --git`` line resets the path until the next ``+++ b/``, so the
    metadata rows a renamed or newly-added file carries between them
    (``index``, ``new file mode``, ``similarity index``, ...) fall into
    no path's count rather than inflating the previous file's past its
    own end."""
    lines, path, number = {}, None, 0
    for row in diff.splitlines():
        if row.startswith("diff --git "):
            path = None
        elif row.startswith("+++ b/"):
            path = row[len("+++ b/"):]
            lines[path] = set()
        elif match := HUNK.match(row):
            number = int(match[1])
        elif path is None or row.startswith("-") or row.startswith("\\"):
            continue
        else:
            lines[path].add(number)
            number += 1
    return lines


def normalised(finding):
    """The finding with an integer line, `None` when it is unreadable."""
    try:
        if isinstance(finding["line"], bool):
            return None
        path, line, comment = (
            finding["path"], int(finding["line"]),
            finding["comment"].strip())
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(path, str) or not comment:
        return None
    return {"path": path, "line": line, "comment": comment}


def moved():
    """The revision the pull request is on now, when it is not the one
    this round reviewed, and `None` while it still is. The base branch
    advancing is not this: a merge base does not move when its target
    gains commits, so the diff — and every line number in it — is the
    same before and after."""
    head = api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}",
               os.environ["GITHUB_TOKEN"])["head"]["sha"]
    return None if head == os.environ["HEAD_SHA"] else head


def counted(number, thing):
    """A number and what it counts, singular when there is one of it."""
    return f"{number} {thing}{'' if number == 1 else 's'}"


def named(paths):
    """A list of paths as one clause."""
    return ", ".join(f"`{path}`" for path in paths)


def uncovered(coverage):
    """What the round could not read whole, said in the review rather
    than in the job's log alone: a changed file past the prompt's budget
    is reviewed from its diff, or not at all, and a reader is owed that
    before taking a clean review for a read one."""
    lines = []
    if coverage.get("degraded"):
        lines.append("Too big for one prompt, reviewed from their diff "
                     f"alone: {named(coverage['degraded'])}.")
    if coverage.get("unreviewed"):
        lines.append("Too big for one prompt even as a diff, not reviewed "
                     f"at all: {named(coverage['unreviewed'])}.")
    return lines


def elsewhere(findings, why):
    """The remarks that go in the body rather than on their own line,
    under the reason they are there."""
    return ["", why] + [f"- `{f['path']}:{f['line']}` — {f['comment']}"
                        for f in findings]


def describe(nth, withheld, unreadable, coverage):
    """The body of one round's review: which round it is, and what it
    could not say inline. The remarks themselves are comments on the
    lines they are about, save those the body has to carry."""
    lines = [f"Style review by `{os.environ['MODEL']}`, round {nth}."]
    if withheld:
        lines.append(f"{counted(withheld, 'further finding')} went past "
                     "the ten-finding cap.")
    if unreadable:
        lines.append(f"{counted(unreadable, 'finding')} could not be read.")
    return "\n".join(lines + uncovered(coverage))


def verdicts(past, given):
    """What became of each remark by its number: this round's answer where
    it is decisive, and otherwise the one an earlier round recorded. A
    remark somebody has accepted or declined does not go back to open
    because a later round forgot it — the file it was about may have left
    the diff since, and a verdict is a thing somebody acted on."""
    kept = dict(past["verdicts"])
    for answer in given:
        try:
            number, verdict = str(int(answer["remark"])), answer["verdict"]
        except (KeyError, TypeError, ValueError):
            continue
        if verdict in history.DECISIVE:
            kept[number] = verdict
    return kept


def answered(numbers, verdicts):
    """The verdicts on the remarks one round made, by their number, a
    remark nobody has answered yet being missing rather than open."""
    return {str(number): verdicts[str(number)]
            for number in numbers if str(number) in verdicts}


def summary(numbers, verdicts):
    """The tally line of one round, `None` when it made no remark: how
    many it made, then what became of them. The total counts the remarks
    still open too — a round answers what it answers, and the reader is
    owed the size of the pile either way. A state nothing is in is left
    out rather than counted at nought, and one that everything is in is
    said of them all rather than counted at the total."""
    given = [verdicts.get(str(number)) for number in numbers]
    if not given:
        return None
    became = [(given.count(verdict), name) for verdict, name in (
        ("accepted", "accepted"), ("declined", "declined"),
        (None, "still open")) if given.count(verdict)]
    made = counted(len(given), "style remark")
    if len(became) > 1:
        return f"{made}: " + " / ".join(
            f"{number} {name}" for number, name in became)
    number, name = became[0]
    return f"{made}: {'all ' if number > 1 else ''}{name}"


def tallied(body, line, verdicts=None):
    """A review body carrying the tally at its foot, in place of whatever
    tally it carried before, or none at all when there is no line: what a
    review said when it was posted is history, only the tally moves. The
    verdicts ride with it, hidden, so the next round reads them back.

    The tally this function itself writes is always the exact trailing
    shape ``history.TALLY_TAIL`` matches, so that is what is stripped —
    a remark that quotes the marker in the body proper, however it is
    placed, is left alone rather than read as one to replace."""
    match = history.TALLY_TAIL.search(body)
    kept = body[:match.start()] if match else body
    if line is None:
        return kept
    return f"{kept}\n\n{history.scoreboard(verdicts or {})}\n{line}"


def rewrite(review, body):
    """Edit a review already posted, doing nothing when it already reads
    that way. A refusal is logged rather than raised: the tally is worth
    less than the round it would take down with it."""
    if body == review["body"]:
        return
    try:
        api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}"
            f"/reviews/{review['id']}", os.environ["APP_TOKEN"],
            {"body": body}, method="PUT")
    except urllib.error.HTTPError as error:
        print(f"Review {review['id']} could not be edited ({error.code}), "
              "leaving it as it stands.")


def record(clean):
    """Tell the workflow whether the diff was clean, when it asks."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as file:
            file.write(f"clean={str(clean).lower()}\n")


def post_review(body, remarks, comments):
    """One round as one review: the record of every remark it makes, the
    body, and a comment on the line of each remark GitHub takes one for.
    A remark left out of ``comments`` is one the body carries instead,
    because its line is nowhere in the diff or because GitHub refused
    the comments outright."""
    api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}"
        f"/reviews", os.environ["APP_TOKEN"], {
            "commit_id": os.environ["HEAD_SHA"], "event": "COMMENT",
            "body": f"{history.stamp(remarks)}\n{body}", "comments": [
                {"path": f["path"], "line": f["line"], "side": "RIGHT",
                 "body": f["comment"]} for f in comments]})


def main():
    ahead = moved()
    if ahead:
        print(f"The pull request moved to {ahead[:8]} while this round ran, "
              "leaving the review to the round that push starts.")
        record(clean=False)
        return
    with open(os.path.join(history.DIRECTORY, "findings.json")) as file:
        answer = json.load(file)
    reported = answer["findings"]
    if not isinstance(reported, list):
        raise ValueError(f"findings should be a list: {reported!r}")
    findings = [f for f in map(normalised, reported) if f is not None]
    unreadable = len(reported) - len(findings)
    if reported and not findings:
        raise ValueError(f"no readable finding in: {reported!r}")
    with open(os.path.join(history.DIRECTORY, "diff.patch")) as file:
        lines = commentable_lines(file.read())
    withheld, findings = len(findings[10:]), findings[:10]
    inline = [f for f in findings if f["line"] in lines.get(f["path"], set())]
    off_diff = [f for f in findings if f not in inline]
    if off_diff:
        print(f"{counted(len(off_diff), 'finding')} sits outside the diff, "
              "in the body: "
              + ", ".join(f"{f['path']}:{f['line']}" for f in off_diff))
    past = history.load()
    merged = verdicts(past, answer.get("verdicts", []))
    coverage = answer.get("coverage", {})
    gaps = uncovered(coverage)
    record(clean=not findings and not gaps)
    if findings or gaps:
        body = describe(
            len(past["rounds"]) + 1, withheld, unreadable, coverage)
        if off_diff:
            body = "\n".join([body] + elsewhere(
                off_diff, "These are about lines the diff does not show, "
                "where GitHub takes no comment:"))
        try:
            post_review(body, findings, inline)
        except urllib.error.HTTPError as error:
            if error.code != 422:
                raise
            print(f"Inline comments rejected ({error.code}), "
                  "posting the remarks in the body.")
            post_review("\n".join([body] + elsewhere(
                inline, "GitHub refused these as inline comments:")),
                findings, [])
    else:
        print("Nothing to say on the diff, posting nothing.")
    for nth, previous in enumerate(past["rounds"], 1):
        numbers = previous["numbers"]
        line = summary(numbers, merged)
        rewrite(previous, tallied(
            previous["body"], line, answered(numbers, merged)))
        if line:
            print(f"Round {nth}: {line}")


if __name__ == "__main__":
    main()
