"""Ask the model for a style review of the diff in one request.

A changed file is any tracked, authored file: a Python module, a marimo
notebook (a ``docs/notebooks/*.md`` file, its code cells fenced as
``python {.marimo}``), or plain prose, config or workflow. Excluded are
generated artefacts nobody hand-writes, filtered out of the diff before
this module runs. Assembles ``prompt.md``, ``STYLE.md``, the package-local
files that the changed Python files import (as context), the remarks the
previous rounds made with the replies they drew, and one listing per
changed file: the whole new file, unified-diff style, every line numbered
by its position in the new file with a leading ``+``/``-`` for one
added/removed since the merge base, falling back to a plain diff (or to no
representation at all) for a file too big to fit the budget. Sends one
chat completion to the OpenAI-compatible gateway at ``BASE_URL`` and
writes the findings, along with a verdict on each past remark, to
``.style-review/findings.json`` for ``post.py`` to post and tally.

The parts go in from the one that never moves to the one that moves every
round, so that two rounds of the same pull request share a prefix and the
gateway can serve it from its cache: the instructions and ``STYLE.md``
first, then the context files, then the past remarks — a numbered list
that only grows at its end — and last the revision under review, which is
new by the very fact that a round is running. The order is what makes the
prompt append-only, so a part moved earlier costs cache on every round
after it.
"""

import ast
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

import history

BUDGET = 400_000
QUOTE = 2_000
TIMEOUT = 600
ATTEMPTS = 2
LANGUAGES = {
    ".py": "python", ".md": "markdown", ".rst": "rst", ".yml": "yaml",
    ".yaml": "yaml", ".toml": "toml", ".json": "json", ".css": "css",
    ".html": "html",
}


def imports(path):
    """The package-local paths imported by a Python file, an empty list
    when this Python cannot parse it, e.g. a marimo notebook."""
    try:
        with open(path) as file:
            tree = ast.parse(file.read())
    except SyntaxError:
        return []
    package, names = os.path.dirname(path).split(os.sep), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = (
                package[:len(package) + 1 - node.level]
                if node.level else [])
            module = ".".join(
                prefix + ([node.module] if node.module else []))
            if module:
                names.add(module)
                names.update(
                    f"{module}.{alias.name}" for alias in node.names)
    paths = (name.replace(".", "/") + suffix for name in sorted(names)
             for suffix in (".py", "/__init__.py"))
    return [path for path in paths if os.path.exists(path)]


def contents(paths, budget, block):
    """Pair each path with its rendered ``block`` while it and the
    ``"\\n\\n"`` separator joining it to its neighbour fit the budget,
    also returning the leftover budget and the paths dropped."""
    kept, dropped = [], []
    for path in paths:
        text = block(path)
        cost = len(text) + 2
        if cost <= budget:
            kept, budget = kept + [(path, text)], budget - cost
        else:
            dropped.append(path)
    return kept, budget, dropped


def annotated(path, base_sha):
    """The whole new-file content of ``path``, unified-diff style: every
    line prefixed by its new-file line number, with a leading ``+`` for
    one added or ``-`` (unnumbered, since it has no new-file line) for
    one removed since ``base_sha``. Gets git's own diff with enough
    context to cover the whole file in one hunk, rather than
    reimplementing the diff itself."""
    diff = subprocess.run(
        ['git', 'diff', '--merge-base', base_sha, '-U100000', '--', path],
        capture_output=True, text=True, check=True).stdout
    body = diff.split("\n@@", 1)[-1].split("\n", 1)[-1]
    lines, number = [], 0
    for row in body.splitlines():
        if row.startswith("\\"):
            continue
        marker, content = row[:1] or ' ', row[1:]
        if marker == '-':
            lines.append(f"  -{content}")
        else:
            number += 1
            lines.append(f"{number} {'+' if marker == '+' else ' '}{content}")
    return "\n".join(lines)


def language(path):
    """The Markdown fence language for a path's own file type, ``text``
    for anything unrecognised."""
    return LANGUAGES.get(os.path.splitext(path)[1], "text")


def fence(body):
    """A backtick fence at least three long, and one longer than any run
    already in ``body`` — an inline code span is no threat, but a
    notebook's own triple-backtick cell fences must never close it
    early."""
    runs = re.findall("`+", body)
    return "`" * max(3, max((len(run) for run in runs), default=0) + 1)


def section(title, path, body):
    ticks = fence(body)
    return f"# {title}: {path}\n\n{ticks}{language(path)}\n{body}{ticks}"


def changed_block(path, base_sha):
    return section("Changed", path, annotated(path, base_sha) + "\n")


def diff_block(path, base_sha):
    """A plain, small-context ``git diff`` of ``path`` since ``base_sha``
    — the compact fallback for a changed file whose full-file
    ``changed_block`` doesn't fit the budget, same idea as ``assemble``
    dropping a context file for size, except a changed file has no
    smaller-still fallback: one that doesn't fit even as a diff is
    reported as unreviewed rather than dropped silently."""
    diff = subprocess.run(
        ['git', 'diff', '--merge-base', base_sha, '--', path],
        capture_output=True, text=True, check=True).stdout
    return section(
        "Changed (diff only, too big for the full file)", path, diff)


def context_block(path):
    with open(path) as file:
        body = file.read().rstrip("\n") + "\n"
    return section("Context (not under review)", path, body)


def literal(text):
    """Somebody's words on one bounded line, as a Python literal: a
    newline or a backtick in them breaks neither the listing they sit in
    nor the fences below, and a long one cannot eat the budget."""
    return repr(text.strip()[:QUOTE])


def past_block(remarks):
    """The remarks of the previous rounds, each under the number its
    verdict refers to. The list only grows at its end, so a round sends
    the list of the round before it unchanged; what each remark drew is
    in the discussion below it, where a new reply lands at the end
    rather than in the middle of this list."""
    lines = ["# Style remarks from the previous rounds", ""]
    lines += [f"{remark['number']}. `{remark['path']}:{remark['line']}` — "
              f"{literal(remark['comment'])}" for remark in remarks]
    return "\n".join(lines)


def discussion_block(transcript):
    """Everything said on the pull request so far, oldest first: the
    replies a remark drew, the reviews of others, the conversation
    around them."""
    ticks = fence(transcript)
    return f"# Discussion so far\n\n{ticks}text\n{transcript}\n{ticks}"


def fitted(text, budget):
    """The text with the budget it leaves, or nothing at all and the
    budget untouched: a part that does not fit is dropped whole rather
    than cut mid-sentence."""
    cost = len(text) + 2
    return (text, budget - cost) if cost <= budget else ("", budget)


def assemble(files, base_sha, past):
    """The one prompt, and what did not fit in it: instructions, style
    guide, context, past remarks, discussion, changes. Every part is
    budgeted as assembled, including the ``"\\n\\n"`` separators the join
    below adds between them, so the request sent to the gateway never
    exceeds ``BUDGET``. The revision under review is budgeted first and
    goes in whole, then the context files, then the past remarks and the
    discussion, however many rounds have piled up: what grew is dropped
    whole rather than evicting a context file, which sits earlier in the
    prompt and would take the prefix of every later round with it. A
    changed file too big for its full-file listing falls back to a plain
    diff of its hunks, and one too big even for that is reported as
    unreviewed rather than raising and crashing the round. What did not
    fit is returned beside the prompt, so that the review can say so
    where it is read rather than in the job's log alone. The notes
    saying as much to the model sit with the changed files they describe
    rather than with the context files: they name whatever did not fit
    this round, so in the prefix they would rewrite its middle every
    time that set changed."""
    deps = sorted(
        {dep for path in files for dep in imports(path)} - set(files))
    with open(".github/style-review/prompt.md") as file:
        instructions = file.read()
    with open("STYLE.md") as file:
        style = f"# STYLE.md\n\n{file.read()}"
    budget = BUDGET + 2 - sum(len(part) + 2 for part in (instructions, style))
    changed, budget, missing = contents(
        files, budget, lambda path: changed_block(path, base_sha))
    degraded, budget, unreviewed = contents(
        missing, budget, lambda path: diff_block(path, base_sha))
    context, budget, dropped = contents(deps, budget, context_block)
    remarks, budget = fitted(
        past_block(past["remarks"]) if past["remarks"] else "", budget)
    if past["remarks"] and not remarks:
        print("The past remarks are past the budget, so this round gives "
              "no verdict.", file=sys.stderr)
    discussion, budget = fitted(
        discussion_block(past["discussion"]) if past["discussion"].strip()
        else "", budget)
    parts = [instructions, style] + [block for _, block in context]
    parts += [part for part in (remarks, discussion) if part]
    notes = []
    if dropped:
        notes.append(f"# Context dropped for size: {', '.join(dropped)}")
    if degraded:
        notes.append(
            "# Changed files past the budget, reviewed from a diff "
            f"only: {', '.join(path for path, _ in degraded)}")
    if unreviewed:
        notes.append(
            "# Changed files past the budget even as a diff, not "
            f"reviewed at all: {', '.join(unreviewed)}")
    for note in notes:
        if len(note) + 2 <= budget:
            parts.append(note)
            budget -= len(note) + 2
    parts += [block for _, block in changed] + [block for _, block in degraded]
    return "\n\n".join(parts), {
        "dropped": dropped, "unreviewed": unreviewed,
        "degraded": [path for path, _ in degraded]}


def complete(request, attempts=ATTEMPTS):
    """The gateway's answer, reading it again when the transfer is cut
    short. A chunked response can end mid-body — an ``IncompleteRead``
    four minutes in left [#661](https://github.com/discopy/discopy/pull/661)
    with no review at all — and a connection reset or a timeout is the
    same failure. An ``HTTPError`` is the gateway answering rather than
    the transfer failing, so it is raised at once, for ``ask`` to read
    the body of, rather than asked again: it is a subclass of
    ``URLError`` and would otherwise be caught below. The attempts are
    capped so that the worst case stays inside the job's own timeout."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except (http.client.IncompleteRead, urllib.error.URLError,
                TimeoutError) as error:
            print(f"gateway transfer failed ({error!r}), attempt {attempt} "
                  f"of {attempts}.", file=sys.stderr)
            if attempt == attempts:
                raise


def ask(prompt):
    """One chat completion. Reasoning is left to the model's own default
    (some gateways mandate it, and quality suffers when it's forced off).
    ``max_tokens`` is 32,768 rather than the previous 8,192 so reasoning
    tokens, which share the same limit, don't crowd out the answer."""
    url = os.environ["BASE_URL"].rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": os.environ["MODEL"], "temperature": 0, "max_tokens": 32_768,
        "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"})
    try:
        body = complete(request)
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")
        print(f"gateway error {error.code}: {text}", file=sys.stderr)
        raise
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        print(f"gateway finish_reason={choice.get('finish_reason')!r} "
              f"usage={body.get('usage')}", file=sys.stderr)
    message = choice["message"]
    answer = message.get("content") or message.get("reasoning") or ""
    if "{" not in answer or "}" not in answer:
        raise ValueError(f"no JSON in the gateway answer: {answer[:200]!r}")
    span = answer[answer.index("{"):answer.rindex("}") + 1]
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        print(f"gateway answer isn't valid JSON: {answer[:3000]!r}",
              file=sys.stderr)
        raise


def main():
    with open(os.path.join(history.DIRECTORY, "files.txt")) as file:
        files = [path for path in file.read().splitlines()
                 if path and os.path.exists(path)]
    past = history.load()
    prompt, coverage = assemble(files, os.environ["BASE_SHA"], past)
    answer = ask(prompt)
    answer["coverage"] = coverage
    with open(os.path.join(history.DIRECTORY, "findings.json"), "w") as file:
        json.dump(answer, file)
    print(f"{len(answer.get('findings', []))} findings, "
          f"{len(answer.get('verdicts', []))} verdicts on "
          f"{len(past['remarks'])} past remarks; "
          f"{len(coverage['degraded'])} changed files read from their diff "
          f"alone, {len(coverage['unreviewed'])} not read at all.")


if __name__ == "__main__":
    main()
