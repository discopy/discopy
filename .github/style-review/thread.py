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
    """One dict per contribution, with a `when` timestamp to sort by, an
    `anchor` of `path` or `path:line` for a comment on the diff, else
    `None`, and the `thread` it belongs to, which is the comment it
    answers or itself: two threads can sit on one line, and it is the
    thread, not the line, whose last word must survive the budget. A
    pending review has no `submitted_at` and is not a contribution yet,
    so it is left out."""
    result = [
        {"when": comment["created_at"], "author": author(comment),
         "anchor": None, "thread": None, "body": comment["body"] or ""}
        for comment in conversation]
    result += [
        {"when": comment["created_at"], "author": author(comment),
         "anchor": anchor(comment), "body": comment["body"] or "",
         "thread": comment.get("in_reply_to_id") or comment["id"]}
        for comment in diff_comments]
    result += [
        {"when": review["submitted_at"], "author": author(review),
         "anchor": None, "thread": None,
         "body": f"[{review['state']}] {review['body']}"}
        for review in reviews
        if review["submitted_at"] and (review["body"] or "").strip()]
    return sorted(result, key=lambda entry: entry["when"])


def block(entry):
    """One transcript entry, as a Markdown heading and its body."""
    head = f"{entry['author']} ({entry['when']})"
    if entry["anchor"]:
        head += f" on {entry['anchor']}"
    return f"### {head}\n\n{entry['body']}"


def note(dropped):
    """What the transcript says of the entries it had to leave out."""
    if not dropped:
        return ""
    return (f"_{dropped} earlier message{'s' if dropped > 1 else ''} "
            "omitted for size._\n\n")


def latest(items):
    """The last entry of each review thread, by identity: the most recent
    word on a flag somebody may still be waiting on."""
    seen, keep = set(), set()
    for entry in reversed(items):
        if entry["thread"] and entry["thread"] not in seen:
            seen.add(entry["thread"])
            keep.add(id(entry))
    return keep


def render(items, budget):
    """The transcript, oldest first. Past the ``budget`` the oldest entry
    goes first, sparing the last word of each thread while any other
    entry is left; the note counts against the budget rather than being
    added on top of it, and the spared entries are cut too rather than
    hand back a transcript over the budget, which the prompt would drop
    whole for less. A budget too small for anything at all gives nothing
    rather than the note alone, which would be over it in its turn."""
    keep = latest(items)
    blocks, dropped = [(entry, block(entry)) for entry in items], 0
    while blocks and sum(
            len(text) + 2 for _, text in blocks) + len(note(dropped)) > budget:
        droppable = [index for index, (entry, _) in enumerate(blocks)
                     if id(entry) not in keep]
        del blocks[droppable[0] if droppable else 0]
        dropped += 1
    if not blocks:
        return ""
    return note(dropped) + "\n\n".join(text for _, text in blocks)
