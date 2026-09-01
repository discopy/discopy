"""Read the style remarks already posted on the pull request.

Every review the style reviewer posts starts with a hidden record of the
remarks it made, so a later round can read them back whole rather than
parsing its own prose. This module collects them in the order the rounds
posted them, each with the replies it received, and writes them to
``.style-review/history.json``: ``review.py`` shows them to the model,
which judges whether each was taken into account, and ``post.py`` writes
each round's tally onto the round that made those remarks.

The same three listings render the discussion so far, which goes in the
same file: everything said on the pull request, ours and everyone else's,
is read once here rather than once per module that wants it. ``DIRECTORY``
is the workspace the workflow's steps hand their files through, named here
for the scripts that read and write them.
"""

import json
import os
import re

import thread
from github import listing

DIRECTORY = ".style-review"
MARKER = "<!-- style-review "
TALLY = "<!-- style-review-tally"
DECISIVE = ("accepted", "declined")


def hidden(marker, payload):
    """An HTML comment carrying JSON, which a review body hides what the
    next round reads in. Escaping ``>`` keeps a payload quoting one from
    closing the comment early, and ``json.loads`` reads the escape
    back."""
    return marker + json.dumps(payload).replace(">", r"\u003e") + " -->"


def stamp(remarks):
    """The hidden record a review carries: the remarks it made."""
    return hidden(MARKER, remarks)


def remarklike(made):
    """Whether a decoded record is a list of remarks, rather than of
    whatever else somebody quoting the marker happened to write. Field
    types are checked here, at the boundary, so a malformed record never
    reaches a caller that assumes a remark's ``comment`` is a string."""
    return isinstance(made, list) and all(
        isinstance(remark, dict)
        and {"path", "line", "comment"} <= remark.keys()
        and isinstance(remark["path"], str)
        and isinstance(remark["line"], int)
        and not isinstance(remark["line"], bool)
        and isinstance(remark["comment"], str) for remark in made)


def recorded(body):
    """The remarks a review recorded, ``None`` when it carries no record
    this module can read, i.e. when it is not one of ours: a body opening
    on the marker and going on with anything else is somebody quoting it,
    never a reason to lose the rounds that follow."""
    if not body.startswith(MARKER):
        return None
    try:
        made = json.loads(body[len(MARKER):body.index(" -->")])
    except ValueError:
        return None
    return made if remarklike(made) else None


def scoreboard(verdicts):
    """The hidden half of a tally: what became of each remark, by its
    number, so that a later round reads back a verdict somebody has
    already acted on rather than asking for it again."""
    return hidden(f"{TALLY} ", verdicts)


TALLY_TAIL = re.compile(
    r"\n\n" + re.escape(TALLY) + r" ([^\n]*) -->\n[^\n]*\Z")


def scored(body):
    """The verdicts a tally carries, empty for a body carrying none or
    carrying one this module cannot read. ``post.tallied`` only ever
    writes one at the very end of the body, so this looks there and
    nowhere else: a remark that quotes the marker in passing, however it
    is placed, is never mistaken for one."""
    if TALLY not in body:
        return {}
    match = TALLY_TAIL.search(body)
    if not match:
        return {}
    try:
        kept = json.loads(match[1])
    except ValueError:
        return {}
    return ({number: verdict for number, verdict in kept.items()
             if verdict in DECISIVE} if isinstance(kept, dict) else {})


def empty():
    """The history of a pull request no round has been posted on yet."""
    return {"rounds": [], "remarks": [], "discussion": "", "verdicts": {}}


def load():
    """The history this module last wrote, empty when it wrote none."""
    path = os.path.join(DIRECTORY, "history.json")
    if not os.path.exists(path):
        return empty()
    with open(path) as file:
        return json.load(file)


def history(repo, number, token, bot):
    """The rounds posted so far, oldest first, each with the remarks it
    made — numbered across every round, since that is what a verdict
    names — and everything said on the pull request as one transcript,
    from the three listings read once. Each round carries the verdicts on
    its own remarks, so they are read back from all of them.

    Only a review posted by ``bot``, the discopy-bot's own login, is read
    as a round: a review from anybody else starting with ``MARKER`` and
    carrying valid JSON is somebody quoting it, not a round to trust —
    nothing about the marker itself proves who wrote it. ``user`` is
    ``None`` for a deleted account, the same as ``thread.author`` already
    guards against, and one is never the bot's own."""
    reviews = listing(f"/repos/{repo}/pulls/{number}/reviews", token)
    comments = listing(f"/repos/{repo}/pulls/{number}/comments", token)
    conversation = listing(f"/repos/{repo}/issues/{number}/comments", token)
    discussion = thread.render(
        thread.entries(conversation, comments, reviews), thread.BUDGET)
    posted = [(review, recorded(review["body"] or "")) for review in reviews
              if review["submitted_at"]
              and (review["user"] or {}).get("login") == bot]
    remarks, rounds, verdicts = [], [], {}
    for review, made in posted:
        if made is None:
            continue
        for remark in made:
            remark["number"] = len(remarks) + 1
            remarks.append(remark)
        rounds.append({
            "id": review["id"], "body": review["body"],
            "numbers": [remark["number"] for remark in made]})
        verdicts.update(scored(review["body"]))
    return {"rounds": rounds, "remarks": remarks, "verdicts": verdicts,
            "discussion": discussion}


def main():
    os.makedirs(DIRECTORY, exist_ok=True)
    past = history(os.environ["REPO"], os.environ["PR_NUMBER"],
                   os.environ["GITHUB_TOKEN"], os.environ["BOT_LOGIN"])
    with open(os.path.join(DIRECTORY, "history.json"), "w") as file:
        json.dump(past, file)
    print(f"{len(past['rounds'])} rounds so far, "
          f"{len(past['remarks'])} remarks, "
          f"{len(past['discussion'])} characters of discussion.")


if __name__ == "__main__":
    main()
