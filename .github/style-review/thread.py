"""Fetch the PR discussion so far and render it as a plain transcript.

Merges chronologically the conversation comments, the review comments
anchored on the diff, and the body of every submitted review, then writes
the result to ``.style-review/thread.md`` for ``review.py`` to fold into
the prompt as its own section. A PR with no discussion yet writes an
empty file.
"""

import json
import os
import urllib.request

DIRECTORY = ".style-review"
API = "https://api.github.com"
BUDGET = 40_000


def get(url, token):
    """Every item of a paginated GitHub API endpoint."""
    items, page = [], 1
    while True:
        request = urllib.request.Request(
            f"{url}{'&' if '?' in url else '?'}page={page}&per_page=100",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = json.load(response)
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def entries(repo, pr_number, token):
    """One dict per contribution, with a `when` timestamp to sort by and
    an `anchor` of `path:line` for a comment on the diff, else `None`."""
    base = f"{API}/repos/{repo}"
    conversation = get(f"{base}/issues/{pr_number}/comments", token)
    diff_comments = get(f"{base}/pulls/{pr_number}/comments", token)
    reviews = get(f"{base}/pulls/{pr_number}/reviews", token)
    result = [
        {"when": comment["created_at"], "author": comment["user"]["login"],
         "anchor": None, "body": comment["body"] or ""}
        for comment in conversation]
    result += [
        {"when": comment["created_at"], "author": comment["user"]["login"],
         "anchor": f"{comment['path']}:"
                   f"{comment.get('line') or comment['original_line']}",
         "body": comment["body"] or ""}
        for comment in diff_comments]
    result += [
        {"when": review["submitted_at"], "author": review["user"]["login"],
         "anchor": None, "body": f"[{review['state']}] {review['body']}"}
        for review in reviews if (review["body"] or "").strip()]
    return sorted(result, key=lambda entry: entry["when"])


def block(entry):
    head = f"{entry['author']} ({entry['when']})"
    if entry["anchor"]:
        head += f" on {entry['anchor']}"
    return f"### {head}\n\n{entry['body']}"


def render(items, budget):
    """The transcript, oldest first. Past the ``budget``, the oldest
    entries are dropped first, except the latest entry on each diff
    anchor: that is the exchange attached to a still-open flag."""
    latest_on_anchor = {
        entry["anchor"] for entry in items if entry["anchor"]}
    seen, protected = set(), set()
    for entry in reversed(items):
        if entry["anchor"] in latest_on_anchor - seen:
            protected.add(id(entry))
            seen.add(entry["anchor"])
    blocks = [(entry, block(entry)) for entry in items]
    dropped = 0
    while sum(len(text) + 2 for _, text in blocks) > budget:
        droppable = [
            i for i, (entry, _) in enumerate(blocks)
            if id(entry) not in protected]
        if not droppable:
            break
        del blocks[droppable[0]]
        dropped += 1
    note = (f"_{dropped} earlier message"
            f"{'s' if dropped > 1 else ''} omitted for size._\n\n"
            if dropped else "")
    return note + "\n\n".join(text for _, text in blocks)


def main():
    repo, pr_number = os.environ["REPO"], os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]
    items = entries(repo, pr_number, token)
    transcript = render(items, BUDGET) if items else ""
    with open(os.path.join(DIRECTORY, "thread.md"), "w") as file:
        file.write(transcript)


if __name__ == "__main__":
    main()
