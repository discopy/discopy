"""Read the style remarks already posted on the pull request.

Every review the style reviewer posts starts with a hidden record of the
remarks it made, so a later round can read them back whole rather than
parsing its own prose. This module collects them in the order the rounds
posted them, each with the replies it received, and writes them to
``.style-review/history.json``: ``review.py`` shows them to the model,
which judges whether each was taken into account, and ``post.py`` writes
the tally of its verdicts onto the newest review of them all, where the
thread is being read.

The same three listings render the discussion so far, which goes in the
same file: everything said on the pull request, ours and everyone else's,
is read once here rather than once per module that wants it.
"""

import json
import os

import thread
from github import listing

DIRECTORY = ".style-review"
"""The workspace the workflow's steps hand their files through, named
here once for the scripts that read and write them."""
MARKER = "<!-- style-review "
TALLY = "<!-- style-review-tally -->"


def stamp(remarks):
    """The hidden record a review carries. Escaping ``>`` keeps a remark
    quoting one from closing the HTML comment early, and ``json.loads``
    reads the escape back."""
    return MARKER + json.dumps(remarks).replace(">", "\\u003e") + " -->"


def recorded(body):
    """The remarks a review recorded, ``None`` when it carries no record
    this module can read, i.e. when it is not one of ours: a body opening
    on the marker and going on with anything else is somebody quoting it,
    never a reason to lose the rounds that follow."""
    if not body.startswith(MARKER):
        return None
    try:
        return json.loads(body[len(MARKER):body.index(" -->")])
    except ValueError:
        return None


def empty():
    """The history of a pull request no round has been posted on yet."""
    return {"rounds": 0, "reviews": [], "remarks": [], "discussion": ""}


def load():
    """The history this module last wrote, empty when it wrote none."""
    path = os.path.join(DIRECTORY, "history.json")
    if not os.path.exists(path):
        return empty()
    with open(path) as file:
        return json.load(file)


def history(repo, number, token):
    """The rounds posted so far with their remarks numbered across all of
    them, and everything said on the pull request as one transcript, from
    the three listings read once."""
    reviews = listing(f"/repos/{repo}/pulls/{number}/reviews", token)
    comments = listing(f"/repos/{repo}/pulls/{number}/comments", token)
    conversation = listing(f"/repos/{repo}/issues/{number}/comments", token)
    discussion = thread.render(
        thread.entries(conversation, comments, reviews), thread.BUDGET)
    rounds = [(review, recorded(review["body"] or "")) for review in reviews]
    rounds = [(review, made) for review, made in rounds if made is not None]
    remarks = []
    for _, made in rounds:
        for remark in made:
            remark["number"] = len(remarks) + 1
            remarks.append(remark)
    return {
        "rounds": len(rounds), "remarks": remarks, "discussion": discussion,
        "reviews": [{"id": review["id"], "body": review["body"]}
                    for review, _ in rounds]}


def main():
    os.makedirs(DIRECTORY, exist_ok=True)
    past = history(os.environ["REPO"], os.environ["PR_NUMBER"],
                   os.environ["GITHUB_TOKEN"])
    with open(os.path.join(DIRECTORY, "history.json"), "w") as file:
        json.dump(past, file)
    print(f"{past['rounds']} rounds so far, {len(past['remarks'])} remarks, "
          f"{len(past['discussion'])} characters of discussion.")


if __name__ == "__main__":
    main()
