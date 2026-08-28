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
next one can read them back. Whatever the model made of those past remarks
— accepted, declined or still open — is tallied onto the newest review of
them all, where the thread is being read: a round that posts carries the
tally in the body it posts, a round with nothing to say edits the newest
review already there, and any older one still carrying a tally is stripped
of it, so there is only ever the one.

A round whose head has moved posts nothing: its findings are about lines
somebody has already replaced, and the push that replaced them starts a
round of its own.
"""

import json
import os
import re
import urllib.error

import history
from github import api

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def commentable_lines(diff):
    """Map each path in a unified diff to the new-file lines it shows."""
    lines, path = {}, None
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[len("+++ b/"):]
            lines[path] = set()
        elif (match := HUNK.match(row)) and path is not None:
            start, length = int(match[1]), int(match[2] or "1")
            lines[path].update(range(start, start + length))
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
    """The verdict on each past remark in the order they were made, `None`
    for one the model skipped, which is one still open."""
    said = {}
    for answer in given:
        try:
            said[int(answer["remark"])] = answer["verdict"]
        except (KeyError, TypeError, ValueError):
            continue
    return [said.get(remark["number"]) for remark in past["remarks"]]


def summary(given):
    """The tally line, `None` before any remark has been made: how many
    were made in all, then what became of them. The total counts the
    remarks still open too — a round answers what it answers, and the
    reader is owed the size of the pile either way — so it says the three
    apart rather than calling them all taken into account."""
    if not given:
        return None
    accepted, declined = given.count("accepted"), given.count("declined")
    waiting = len(given) - accepted - declined
    line = (f"{counted(len(given), 'style remark')}: "
            f"{accepted} accepted / {declined} declined")
    return line if not waiting else f"{line} / {waiting} still open"


def tallied(body, line):
    """A review body carrying the tally at its foot, in place of whatever
    tally it carried before, or none at all when there is no line: what a
    review said when it was posted is history, only the tally moves."""
    kept = body.rsplit(f"\n\n{history.TALLY}\n", 1)[0].rstrip()
    return kept if line is None else f"{kept}\n\n{history.TALLY}\n{line}"


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
    line = summary(verdicts(past, answer.get("verdicts", [])))
    record(clean=not findings)
    carried = past["reviews"]
    if findings:
        body = describe(
            past["rounds"] + 1, len(off_diff), withheld, unreadable)
        try:
            post_review(tallied(body, line), findings)
        except urllib.error.HTTPError as error:
            print(f"Inline comments rejected ({error.code}), "
                  "posting the remarks in the body.")
            post_review(tallied("\n".join(
                [body, "", "GitHub refused these as inline comments:"] + [
                    f"- `{f['path']}:{f['line']}` — {f['comment']}"
                    for f in findings]), line), findings, inline=False)
    else:
        print("Nothing to say on the diff, posting nothing.")
        if carried:
            newest = carried[-1]
            rewrite(newest, tallied(newest["body"], line))
            carried = carried[:-1]
    for review in carried:
        rewrite(review, tallied(review["body"], None))
    if line:
        print(line)


if __name__ == "__main__":
    main()
