"""Open the codebase-review issue from the files the read wrote.

Reads ``.codebase-review/report.md`` and ``.codebase-review/bugs.md``, opens a
``codebase-review``-labelled issue with the report as its body — as the GitHub
App authenticated by ``APP_TOKEN`` — and posts the bugs as its first
comment. A missing or empty report fails the run rather than post an empty
read; a missing bugs file posts no comment. A rerun on the same day updates
the day's open issue instead of opening a twin.
"""

import datetime
import json
import os
import urllib.error
import urllib.request

DIRECTORY = ".codebase-review"
LABEL = "codebase-review"


def request(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {
        "Authorization": f"Bearer {os.environ['APP_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def ensure_label(api):
    try:
        request(f"{api}/labels", {
            "name": LABEL, "color": "1D76DB",
            "description": "Findings from a full read of the codebase"})
    except urllib.error.HTTPError as error:
        if error.code != 422:
            raise


def read(name):
    try:
        with open(os.path.join(DIRECTORY, name)) as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""


def main():
    report, bugs = read("report.md"), read("bugs.md")
    if not report:
        raise ValueError("the read wrote no report")
    api = f"https://api.github.com/repos/{os.environ['REPO']}"
    ensure_label(api)
    date = datetime.datetime.now(datetime.timezone.utc).date()
    title = f"Findings from a full read of the codebase ({date})"
    body = (f"{report}\n\n*Posted by the"
            f" [codebase-review run]({os.environ['RUN_URL']}).*")
    issues = request(f"{api}/issues?labels={LABEL}&state=open&per_page=100")
    reused = [i for i in issues
              if "pull_request" not in i and i["title"] == title]
    if reused:
        issue = reused[0]
        request(issue["url"], {"body": body}, method="PATCH")
    else:
        issue = request(f"{api}/issues", {
            "title": title, "labels": [LABEL], "body": body})
    if bugs:
        newest = (request(
            f"{api}/issues/{issue['number']}/comments"
            "?per_page=1&sort=created&direction=desc") if reused else [])
        if newest:
            request(newest[0]["url"], {"body": bugs}, method="PATCH")
        else:
            request(
                f"{api}/issues/{issue['number']}/comments", {"body": bugs})
    print(f"Opened {issue['html_url']}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as file:
            file.write(f"Opened {issue['html_url']}\n")


if __name__ == "__main__":
    main()
