"""Post the style reviewer's findings as one pull request review.

Reads ``.style-review/findings.json`` and ``.style-review/diff.patch``,
posts the findings sitting on a line of the diff as inline comments and the
others in the review body, as the GitHub App authenticated by ``APP_TOKEN``.
A clean run posts nothing.
"""

import json
import os
import re
import urllib.error
import urllib.request

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


def describe(findings, withheld):
    lines = [f"Style review by `{os.environ['MODEL']}`."] + [
        f"- `{f['path']}:{f['line']}` — {f['comment']}" for f in findings]
    if withheld:
        lines.append(f"…and {withheld} more past the ten-finding cap.")
    return "\n".join(lines)


def record(clean):
    """Tell the workflow whether the diff was clean, when it asks."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as file:
            file.write(f"clean={str(clean).lower()}\n")


def post_review(body, inline):
    url = (f"https://api.github.com/repos/{os.environ['REPO']}"
           f"/pulls/{os.environ['PR_NUMBER']}/reviews")
    payload = {
        "commit_id": os.environ["HEAD_SHA"], "event": "COMMENT",
        "body": body, "comments": [
            {"path": f["path"], "line": f["line"], "side": "RIGHT",
             "body": f["comment"]} for f in inline]}
    request = urllib.request.Request(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {os.environ['APP_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=60).close()


def main():
    with open(os.path.join(DIRECTORY, "findings.json")) as file:
        reported = json.load(file)["findings"]
    if not isinstance(reported, list):
        raise ValueError(f"findings should be a list: {reported!r}")
    findings = [normalised(f) for f in reported]
    if None in findings:
        unreadable = [f for f, n in zip(reported, findings) if n is None]
        raise ValueError(f"unreadable findings: {unreadable!r}")
    withheld, findings = len(findings[10:]), findings[:10]
    record(clean=not findings)
    if not findings:
        print("The diff is clean, posting nothing.")
        return
    with open(os.path.join(DIRECTORY, "diff.patch")) as file:
        lines = commentable_lines(file.read())
    inline = [
        f for f in findings if f["line"] in lines.get(f["path"], set())]
    outline = [
        f for f in findings if f["line"] not in lines.get(f["path"], set())]
    try:
        post_review(describe(outline, withheld), inline)
    except urllib.error.HTTPError as error:
        print(f"Inline comments rejected ({error.code}), "
              "posting all in the body.")
        post_review(describe(findings, withheld), inline=[])


if __name__ == "__main__":
    main()
