"""Render the PR discussion so far as a plain transcript.

Merges chronologically the conversation comments, the review comments
anchored on the diff, and the body of every submitted review, so that a
round reads the replies its remarks drew rather than re-litigating them.
``history.py`` fetches those three listings — for the tally as well as
for this — and folds the result into ``.style-review/history.json``.
"""

BUDGET = 40_000


def author(item):
    """An item's author login, `"ghost"` for a deleted account."""
    return (item["user"] or {}).get("login", "ghost")


def anchor(comment):
    """A diff comment's `path` or `path:line`, `line` falling back to
    `original_line` when the comment is outdated, absent when neither is
    set, e.g. a comment on the file as a whole."""
    line = comment.get("line") or comment.get("original_line")
    return f"{comment['path']}:{line}" if line else comment["path"]


def entries(conversation, diff_comments, reviews):
    """One dict per contribution, with a `when` timestamp to sort by and
    an `anchor` of `path` or `path:line` for a comment on the diff, else
    `None`. A pending review has no `submitted_at` and is not a
    contribution yet, so it is left out."""
    result = [
        {"when": comment["created_at"], "author": author(comment),
         "anchor": None, "body": comment["body"] or ""}
        for comment in conversation]
    result += [
        {"when": comment["created_at"], "author": author(comment),
         "anchor": anchor(comment), "body": comment["body"] or ""}
        for comment in diff_comments]
    result += [
        {"when": review["submitted_at"], "author": author(review),
         "anchor": None, "body": f"[{review['state']}] {review['body']}"}
        for review in reviews
        if review["submitted_at"] and (review["body"] or "").strip()]
    return sorted(result, key=lambda entry: entry["when"])


def block(entry):
    """One transcript entry, as a Markdown heading and its body."""
    head = f"{entry['author']} ({entry['when']})"
    if entry["anchor"]:
        head += f" on {entry['anchor']}"
    return f"### {head}\n\n{entry['body']}"


def render(items, budget):
    """The transcript, oldest first. Past the ``budget``, the oldest
    entries are dropped first, except the latest entry on each diff
    anchor: the most recent comment in a still-open flag's thread."""
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
