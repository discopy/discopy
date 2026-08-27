"""Post the style reviewer's findings as one pull request review.

Reads ``.style-review/findings.json`` and ``.style-review/diff.patch``,
posts the findings sitting on a line of the diff as inline comments and the
others in the review body, as the GitHub App authenticated by ``APP_TOKEN``.
A clean run posts nothing.

One review per round, and the round records the remarks it made so that the
next one can read them back. Whatever the model made of those past remarks
— accepted, declined or still open — is tallied onto the body of the first
review, so the top of the thread says how the review is landing rather than
the reader counting the rounds.
"""

import json
import os
import re
import urllib.error

import history
from github import api

DIRECTORY = ".style-review"
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


def describe(nth, findings, withheld, unreadable):
    lines = [f"Style review by `{os.environ['MODEL']}`, round {nth}."] + [
        f"- `{f['path']}:{f['line']}` — {f['comment']}" for f in findings]
    if withheld:
        lines.append(f"…and {withheld} more past the ten-finding cap.")
    if unreadable:
        lines.append(f"…and {unreadable} unreadable findings dropped.")
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
    """The tally line, `None` when no remark has been answered yet: a
    review whose remarks are all still open says nothing about them."""
    accepted, declined = given.count("accepted"), given.count("declined")
    waiting = len(given) - accepted - declined
    if not accepted + declined:
        return None
    line = (f"{accepted}+{declined} style remarks taken into account: "
            f"{accepted} accepted / {declined} declined")
    return line if not waiting else f"{line}, {waiting} still open"


def retally(past, given):
    """Edit the tally onto the first review, below whatever it said when
    it was posted: its own remarks are history, only the tally moves."""
    line = summary(verdicts(past, given))
    if past["first"] is None or line is None:
        return
    body = past["first"]["body"].rsplit(
        f"\n\n{history.TALLY}\n", 1)[0].rstrip()
    try:
        api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}"
            f"/reviews/{past['first']['id']}", os.environ["APP_TOKEN"],
            {"body": f"{body}\n\n{history.TALLY}\n{line}"}, method="PUT")
    except urllib.error.HTTPError as error:
        print(f"The tally was refused ({error.code}), leaving the first "
              "review as it stands.")
        return
    print(line)


def record(clean):
    """Tell the workflow whether the diff was clean, when it asks."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as file:
            file.write(f"clean={str(clean).lower()}\n")


def post_review(body, inline, remarks):
    api(f"/repos/{os.environ['REPO']}/pulls/{os.environ['PR_NUMBER']}"
        f"/reviews", os.environ["APP_TOKEN"], {
            "commit_id": os.environ["HEAD_SHA"], "event": "COMMENT",
            "body": f"{history.stamp(remarks)}\n{body}", "comments": [
                {"path": f["path"], "line": f["line"], "side": "RIGHT",
                 "body": f["comment"]} for f in inline]})


def main():
    with open(os.path.join(DIRECTORY, "findings.json")) as file:
        answer = json.load(file)
    reported = answer["findings"]
    if not isinstance(reported, list):
        raise ValueError(f"findings should be a list: {reported!r}")
    findings = [f for f in map(normalised, reported) if f is not None]
    unreadable = len(reported) - len(findings)
    if reported and not findings:
        raise ValueError(f"no readable finding in: {reported!r}")
    withheld, findings = len(findings[10:]), findings[:10]
    past = history.load()
    retally(past, answer.get("verdicts", []))
    record(clean=not findings)
    if not findings:
        print("The diff is clean, posting nothing.")
        return
    nth = past["rounds"] + 1
    with open(os.path.join(DIRECTORY, "diff.patch")) as file:
        lines = commentable_lines(file.read())
    inline = [
        f for f in findings if f["line"] in lines.get(f["path"], set())]
    outline = [
        f for f in findings if f["line"] not in lines.get(f["path"], set())]
    try:
        post_review(
            describe(nth, outline, withheld, unreadable), inline, findings)
    except urllib.error.HTTPError as error:
        print(f"Inline comments rejected ({error.code}), "
              "posting all in the body.")
        post_review(describe(nth, findings, withheld, unreadable),
                    inline=[], remarks=findings)


if __name__ == "__main__":
    main()
