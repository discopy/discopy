"""The slice of the GitHub REST API the style reviewer needs.

Two callers with two identities: ``history.py`` reads the previous rounds
with the workflow token, ``post.py`` writes the review and edits the first
one with the discopy-bot token, so the token is a parameter rather than
read from the environment here.
"""

import json
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
SIZE = 100


def api(path, token, payload=None, method=None):
    """One JSON request, its parsed answer or ``None`` for an empty body.
    Prints the response body of a failure before re-raising it, so that a
    422 says which field GitHub rejected."""
    request = urllib.request.Request(
        API + path, None if payload is None else json.dumps(payload).encode(),
        {"Authorization": f"Bearer {token}",
         "Accept": "application/vnd.github+json",
         "Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")
        print(f"github error {error.code} on {path}: {text}", file=sys.stderr)
        raise
    return json.loads(body) if body else None


def listing(path, token):
    """Every item of a paginated listing, oldest first as GitHub gives
    them, following the pages by their length rather than parsing the
    ``Link`` header."""
    items, page = [], 1
    while True:
        joint = "&" if "?" in path else "?"
        batch = api(f"{path}{joint}per_page={SIZE}&page={page}", token)
        items += batch
        if len(batch) < SIZE:
            return items
        page += 1
