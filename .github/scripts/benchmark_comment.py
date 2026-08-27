"""Post the benchmark comparison as one comment on the pull request.

Runs from ``benchmark-comment.yml``, on the ``workflow_run`` of a finished
benchmark, with the artifact that run staged in ``benchmark-comment``: the
metadata naming the pull request and the three commits, and the comparison
rendered by ``benchmark/report.py``. A ``workflow_run`` carries the
privileges of the default branch and the artifact comes from a run that
does not, so every field of the metadata is checked against the run and
the pull request itself before anything is posted. One comment per pull
request, updated in place.
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request

API = "https://api.github.com"
DIRECTORY = "benchmark-comment"
MARKER = "<!-- discopy-benchmark -->"
LIMIT = 65536
SHA = re.compile("^[0-9a-f]{40}$")


def fail(message):
    print(f"::error::{message}")
    sys.exit(1)


def call(url, method="GET", payload=None):
    """One authenticated call to the REST API, decoded."""
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, body, {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read()
    return json.loads(text) if text else None


def paginate(url):
    """Every page of a listing, a hundred items at a time."""
    items, page = [], 1
    while True:
        batch = call(f"{url}?per_page=100&page={page}")
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def staged(repository, run_id):
    """Whether the run uploaded the artifact at all. What tells a run that
    staged nothing -- cancelled before its first step -- from one whose
    artifact failed to download, which must not pass for silence."""
    artifacts = call(
        f"{API}/repos/{repository}/actions/runs/{run_id}/artifacts"
        "?per_page=100")["artifacts"]
    return any(artifact["name"] == DIRECTORY for artifact in artifacts)


def merge_base(repository, base, head):
    """The merge base of two commits, as the API computes it."""
    return call(f"{API}/repos/{repository}/compare/{base}...{head}"
                )["merge_base_commit"]["sha"]


def candidates(repository, run):
    """The open pull requests for the run's head, when the run itself
    lists none: a run from a fork does not carry its pull requests."""
    source = (run.get("head_repository") or {}).get("full_name") or ""
    owner, branch = source.split("/")[0], run.get("head_branch") or ""
    query = urllib.parse.quote(f"{owner}:{branch}", safe=":")
    return call(f"{API}/repos/{repository}/pulls?head={query}&state=open")


def unreadable(data, run):
    """Why the metadata is not a description of this run, ``None`` when it
    is. Checked before the pull request number reaches a URL. The artifact
    is written by a run with fewer privileges than this workflow, so it is
    evidence, never authority."""
    number = data.get("pull_request")
    if (not isinstance(number, int) or isinstance(number, bool)
            or data.get("run_id") != run["id"]
            or not all(SHA.match(str(data.get(name, "")))
                       for name in ("base", "previous", "head"))
            or data["head"] != run["head_sha"]):
        return "Invalid benchmark metadata."
    return None


def mismatch(data, run, pull, repository):
    """Why the metadata does not describe this pull request, ``None`` when
    it does."""
    source = (run.get("head_repository") or {}).get("full_name")
    if (pull["base"]["repo"]["full_name"] != repository
            or pull["head"]["repo"]["full_name"] != source
            or pull["head"]["ref"] != run["head_branch"]):
        return "Benchmark metadata does not match its source PR."
    return None


def unattested(data, run, open_pulls):
    """Why the pull request the metadata names is not the one this run
    belongs to, ``None`` when it is. A run from this repository lists its
    pull requests and the named one must be among them, on the same two
    commits. A run from a fork lists none, and ``open_pulls`` -- the open
    pull requests for its head -- must then be exactly the one named,
    since a head with two of them names neither."""
    listed = run.get("pull_requests") or []
    if listed:
        for candidate in listed:
            if candidate["number"] != data["pull_request"]:
                continue
            if (candidate["head"]["sha"] != data["head"]
                    or candidate["base"]["sha"] != data["base"]):
                return "Benchmark run does not belong to this PR."
            return None
        return "Benchmark run does not belong to this PR."
    if [pull["number"] for pull in open_pulls] != [data["pull_request"]]:
        return "Benchmark run does not name the one pull request for its head."
    return None


def sanitised(report):
    """The report with HTML and mentions neutralised: a pull request owns
    the case names ``benchmark/report.py`` prints back."""
    return report.replace("<", "&lt;").replace("@", "&#64;")


def caveats(data, run, pull):
    if pull["base"]["sha"] != data["base"]:
        yield ("The pull request base has changed since this run; rerun the "
               "benchmark for a current comparison.")
    if run["conclusion"] != "success":
        yield f"The benchmark workflow concluded `{run['conclusion']}`."


def body(data, run, pull, report):
    """The whole comment: the marker that identifies it, the comparison,
    what was compared against what, and any caveat on top."""
    def link(repo, sha):
        return f"[`{sha[:7]}`]({repo['html_url']}/commit/{sha})"
    context = (
        f"Comparing head {link(pull['head']['repo'], data['head'])} with its "
        f"merge base {link(pull['base']['repo'], data['previous'])} on the "
        f"same runner. [Workflow run]({run['html_url']}).")
    notes = list(caveats(data, run, pull))
    warning = f"\n\n> [!WARNING]\n> {' '.join(notes)}" if notes else ""
    return f"{MARKER}\n{report}\n{context}{warning}"


def ours(comments):
    """The comment this workflow posted last time, ``None`` on the first.
    The newest of them when there are several, so that duplicates leave
    the stale ones behind rather than the live one."""
    marked = [
        comment for comment in comments
        if (comment.get("user") or {}).get("login") == "github-actions[bot]"
        and (comment.get("body") or "").startswith(MARKER)]
    return max(marked, default=None, key=lambda comment: (
        comment.get("created_at", ""), comment.get("id", 0)))


def comparison():
    path = os.path.join(DIRECTORY, "comparison.md")
    if not os.path.exists(path):
        return ("## Benchmark comparison\n\n**ERROR:** The benchmark run did "
                "not produce a comparison.")
    with open(path) as file:
        return sanitised(file.read())


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as file:
        run = json.load(file)["workflow_run"]
    repository = os.environ["GITHUB_REPOSITORY"]
    path = os.path.join(DIRECTORY, "metadata.json")
    if not os.path.exists(path):
        if staged(repository, run["id"]):
            fail("The benchmark artifact was staged but not downloaded.")
        print("::notice::The benchmark run staged no comparison.")
        return
    with open(path) as file:
        data = json.load(file)
    if reason := unreadable(data, run):
        fail(reason)
    open_pulls = [] if run.get("pull_requests") else candidates(
        repository, run)
    if reason := unattested(data, run, open_pulls):
        fail(reason)
    pull = call(f"{API}/repos/{repository}/pulls/{data['pull_request']}")
    if reason := mismatch(data, run, pull, repository):
        fail(reason)
    if pull["head"]["sha"] != data["head"]:
        print("::notice::Skipping a comparison for a superseded head commit.")
        return
    # `previous` is what the comment links as the merge base, and it comes
    # from an artifact the pull request can write. Both ends of the compare
    # are trusted by here, and a compare of two commits does not move, so
    # the answer is the merge base the benchmark measured against.
    if merge_base(repository, data["base"], data["head"]) != data["previous"]:
        fail("Benchmark metadata does not match its merge base.")
    text = body(data, run, pull, comparison())
    if len(text) > LIMIT:
        fail("Benchmark comparison exceeds GitHub's comment limit.")
    issues = f"{API}/repos/{repository}/issues/{data['pull_request']}"
    if previous := ours(paginate(f"{issues}/comments")):
        call(f"{API}/repos/{repository}/issues/comments/{previous['id']}",
             "PATCH", {"body": text})
    else:
        call(f"{issues}/comments", "POST", {"body": text})


if __name__ == "__main__":
    main()
