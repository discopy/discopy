"""Read the style remarks already posted on the pull request.

Every review the style reviewer posts starts with a hidden record of the
remarks it made, so a later round can read them back whole rather than
parsing its own prose. This module collects them in the order the rounds
posted them, each with the replies it received, and writes them to
``.style-review/history.json``: ``review.py`` shows them to the model,
which judges whether each was taken into account, and ``post.py`` writes
the tally of its verdicts onto the newest review of them all, where the
thread is being read.
"""

import json
import os

from github import listing

DIRECTORY = ".style-review"
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


def human(comment):
    """Whether a comment was written by a person: a verdict reads the
    replies a remark drew, and the reviewer's own posts are not replies."""
    return comment["user"]["type"] != "Bot"


def spoken(comment):
    """Who said it and what they said, the part of a comment a verdict
    reads."""
    return {"author": comment["user"]["login"], "body": comment["body"]}


def replies(comments, remark, review):
    """What a remark drew, when it was posted inline: the people who
    answered the thread it opened. A remark that went to the review body
    has no thread and draws its answers in the conversation."""
    thread = [comment for comment in comments
              if comment["pull_request_review_id"] == review
              and comment["path"] == remark["path"]
              and comment.get("line") == remark["line"]]
    if not thread:
        return []
    return [spoken(comment) for comment in comments
            if comment.get("in_reply_to_id") == thread[0]["id"]
            and human(comment)]


def empty():
    """The history of a pull request no round has been posted on yet."""
    return {"rounds": 0, "reviews": [], "remarks": [], "comments": []}


def load():
    """The history this module last wrote, empty when it wrote none."""
    path = os.path.join(DIRECTORY, "history.json")
    if not os.path.exists(path):
        return empty()
    with open(path) as file:
        return json.load(file)


def history(repo, number, token):
    """The rounds posted so far, their remarks numbered across all of
    them, and the conversation since the first one."""
    rounds = [
        (review, recorded(review["body"] or ""))
        for review in listing(f"/repos/{repo}/pulls/{number}/reviews", token)]
    rounds = [(review, made) for review, made in rounds if made is not None]
    if not rounds:
        return empty()
    comments = listing(f"/repos/{repo}/pulls/{number}/comments", token)
    remarks = []
    for review, made in rounds:
        for remark in made:
            remark["number"] = len(remarks) + 1
            remark["replies"] = replies(comments, remark, review["id"])
            remarks.append(remark)
    since = rounds[0][0]["submitted_at"]
    return {
        "rounds": len(rounds), "remarks": remarks,
        "reviews": [{"id": review["id"], "body": review["body"]}
                    for review, _ in rounds],
        "comments": [
            spoken(comment) for comment in listing(
                f"/repos/{repo}/issues/{number}/comments", token)
            if human(comment) and comment["created_at"] > since]}


def main():
    os.makedirs(DIRECTORY, exist_ok=True)
    past = history(os.environ["REPO"], os.environ["PR_NUMBER"],
                   os.environ["GITHUB_TOKEN"])
    with open(os.path.join(DIRECTORY, "history.json"), "w") as file:
        json.dump(past, file)
    print(f"{past['rounds']} rounds so far, {len(past['remarks'])} remarks, "
          f"{len(past['comments'])} replies since the first.")


if __name__ == "__main__":
    main()
