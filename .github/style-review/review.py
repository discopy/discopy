"""Ask the model for a style review of the diff in one request.

A changed file is any tracked, authored file: a Python module, a marimo
notebook (a ``docs/notebooks/*.md`` file, its code cells fenced as
``python {.marimo}``), or plain prose, config or workflow. Excluded are
generated artefacts nobody hand-writes, filtered out of the diff before
this module runs. Assembles ``prompt.md``, ``STYLE.md``, the PR discussion
so far from ``thread.py`` (when there is one), the package-local files
that the changed Python files import (as context), and one listing per
changed file: the whole new file, unified-diff style, every line numbered
by its position in the new file with a leading ``+``/``-`` for one
added/removed since the merge base. Sends one chat completion to the
OpenAI-compatible gateway at ``BASE_URL`` and writes the findings to
``.style-review/findings.json`` for ``post.py`` to post.
"""

import ast
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

DIRECTORY = ".style-review"
BUDGET = 400_000
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


def context_block(path):
    with open(path) as file:
        body = file.read().rstrip("\n") + "\n"
    return section("Context (not under review)", path, body)


def assemble(files, base_sha, thread=""):
    """The one prompt: instructions, style guide, discussion so far,
    context, changes. Every part is budgeted as assembled, including the
    ``"\\n\\n"`` separators the join below adds between them, so the
    request sent to the gateway never exceeds ``BUDGET``. The changed
    files are the one part that must fit, so they are budgeted right
    after the mandatory instructions and style guide; the discussion, like
    the context, is dropped rather than raising when it does not fit."""
    deps = sorted(
        {dep for path in files for dep in imports(path)} - set(files))
    with open(".github/style-review/prompt.md") as file:
        instructions = file.read()
    with open("STYLE.md") as file:
        style = f"# STYLE.md\n\n{file.read()}"
    budget = BUDGET + 2 - sum(len(part) + 2 for part in (instructions, style))
    changed, budget, missing = contents(
        files, budget, lambda path: changed_block(path, base_sha))
    if missing:
        raise ValueError(f"changed files past the budget: {missing}")
    thread_part = ""
    if thread.strip():
        ticks = fence(thread)
        thread_part = f"# Discussion so far\n\n{ticks}text\n{thread}\n{ticks}"
        if len(thread_part) + 2 > budget:
            thread_part = ""
        else:
            budget -= len(thread_part) + 2
    context, budget, dropped = contents(deps, budget, context_block)
    parts = [instructions, style]
    if thread_part:
        parts.append(thread_part)
    parts += [block for _, block in context]
    if dropped:
        note = f"# Context dropped for size: {', '.join(dropped)}"
        if len(note) + 2 <= budget:
            parts.append(note)
    parts += [block for _, block in changed]
    return "\n\n".join(parts)


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
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
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
    with open(os.path.join(DIRECTORY, "files.txt")) as file:
        files = [path for path in file.read().splitlines()
                 if path and os.path.exists(path)]
    thread_path = os.path.join(DIRECTORY, "thread.md")
    thread = ""
    if os.path.exists(thread_path):
        with open(thread_path) as file:
            thread = file.read()
    findings = ask(assemble(files, os.environ["BASE_SHA"], thread))
    with open(os.path.join(DIRECTORY, "findings.json"), "w") as file:
        json.dump(findings, file)
    print(f"{len(findings.get('findings', []))} findings.")


if __name__ == "__main__":
    main()
