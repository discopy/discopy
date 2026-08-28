"""Post the style reviewer's findings as one pull request review.

Reads ``.style-review/findings.json`` and ``.style-review/diff.patch``,
posts the findings sitting on a line of the diff as inline comments and the
others in the review body, as the GitHub App authenticated by ``APP_TOKEN``.
A clean run posts nothing.

Every remark is an inline comment on the line it is about: a finding that
sits on no line of the diff is dropped rather than moved to the body,
which carries the round and what it could not say rather than a list of
findings. A review of the file at large is not what the diff asked for
([#673](https://github.com/discopy/discopy/issues/673)). The one body
that does list them is the one GitHub leaves when it refuses the inline
comments, where the choice is that shape or no remarks at all.

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
    """Map each path in a unified diff to the lines it adds. A hunk shows
    the lines around a change as well, and GitHub would take a comment on
    those, but the prompt asks for a finding on a line the diff adds, so
    that is what a finding is held to."""
    lines, path, number = {}, None, 0
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[len("+++ b/"):]
            lines[path] = set()
        elif match := HUNK.match(row):
            number = int(match[1])
        elif path is None or row.startswith("-") or row.startswith("\\"):
            continue
        elif row.startswith("+"):
            lines[path].add(number)
            number += 1
        else:
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


def describe(nth, dropped, withheld, unreadable):
    """The body of one round's review: which round it is, and what it
    could not say. The remarks themselves are inline comments on the
    lines they are about, so no list of them belongs here."""
    lines = [f"Style review by `{os.environ['MODEL']}`, round {nth}."]
    if dropped:
        lines.append(
            f"{counted(dropped, 'finding')} sat on no line of the diff.")
    if withheld:
        lines.append(f"{counted(withheld, 'further finding')} went past "
                     "the ten-finding cap.")
    if unreadable:
        lines.append(f"{counted(unreadable, 'finding')} could not be read.")
    return "\n".join(lines)


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
    verdicts ride with it, hidden, so the next round reads them back."""
    kept = body.rsplit(f"\n\n{history.TALLY}", 1)[0].rstrip()
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


def post_review(body, findings, inline=True):
    """One round as one review: the record of the remarks it makes, the
    body, and each remark as a comment on its own line. ``inline`` goes
    false only where GitHub has refused those comments, the remarks then
    going in the body so that a rejection costs the round its shape
    rather than its findings."""
    api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}"
        f"/reviews", os.environ["APP_TOKEN"], {
            "commit_id": os.environ["HEAD_SHA"], "event": "COMMENT",
            "body": f"{history.stamp(findings)}\n{body}", "comments": [
                {"path": f["path"], "line": f["line"], "side": "RIGHT",
                 "body": f["comment"]} for f in findings] if inline else []})


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
    on_diff = [
        f for f in findings if f["line"] in lines.get(f["path"], set())]
    off_diff = [f"{f['path']}:{f['line']}"
                for f in findings if f not in on_diff]
    if off_diff:
        print(f"{counted(len(off_diff), 'finding')} sat on no line of the "
              f"diff, dropped: {', '.join(off_diff)}")
    withheld, findings = len(on_diff[10:]), on_diff[:10]
    past = history.load()
    merged = verdicts(past, answer.get("verdicts", []))
    record(clean=not findings)
    if findings:
        body = describe(
            len(past["rounds"]) + 1, len(off_diff), withheld, unreadable)
        try:
            post_review(body, findings)
        except urllib.error.HTTPError as error:
            if error.code != 422:
                raise
            print(f"Inline comments rejected ({error.code}), "
                  "posting the remarks in the body.")
            post_review("\n".join(
                [body, "", "GitHub refused these as inline comments:"] + [
                    f"- `{f['path']}:{f['line']}` — {f['comment']}"
                    for f in findings]), findings, inline=False)
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
