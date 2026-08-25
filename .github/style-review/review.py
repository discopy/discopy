"""Ask the model for a style review of the diff in one request.

Assembles ``prompt.md``, ``STYLE.md``, the package-local files that the
changed files import (as context), the full text of every changed file with
line numbers and the diff from ``.style-review``, sends one chat completion
to the OpenAI-compatible gateway at ``BASE_URL`` and writes the findings to
``.style-review/findings.json`` for ``post.py`` to post.
"""

import ast
import json
import os
import sys
import urllib.error
import urllib.request

DIRECTORY = ".style-review"
BUDGET = 400_000


def imports(path):
    """The package-local paths imported by a Python file, an empty list
    when this Python cannot parse it."""
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
        with open(path) as file:
            text = block(path, file.read())
        cost = len(text) + 2
        if cost <= budget:
            kept, budget = kept + [(path, text)], budget - cost
        else:
            dropped.append(path)
    return kept, budget, dropped


def numbered(text):
    return "\n".join(
        f"{n} {line}" for n, line in enumerate(text.splitlines(), 1))


def section(title, path, body):
    return f"# {title}: {path}\n\n```python\n{body}```"


def changed_block(path, text):
    return section("Changed", path, numbered(text) + "\n")


def context_block(path, text):
    return section("Context (not under review)", path, text)


def assemble(files, diff):
    """The one prompt: instructions, style guide, context, changes, diff.
    Every part is budgeted as assembled, including the ``"\\n\\n"``
    separators the join below adds between them, so the request sent to
    the gateway never exceeds ``BUDGET``. A changed file too big to fit
    its full text falls back to the diff alone, same as a context file
    dropped for size — except the diff itself may also be truncated for
    size, in which case a file whose hunk falls in the truncated tail
    gets no representation at all, and is reported as such rather than
    as reviewed from a diff it isn't in."""
    deps = sorted(
        {dep for path in files for dep in imports(path)} - set(files))
    if len(diff) > BUDGET // 2:
        diff = diff[:BUDGET // 2] + "\n[diff truncated for size]"
    diff_part = f"# Diff\n\n```diff\n{diff}```"
    with open(".github/style-review/prompt.md") as file:
        instructions = file.read()
    with open("STYLE.md") as file:
        style = f"# STYLE.md\n\n{file.read()}"
    budget = BUDGET + 2 - sum(
        len(part) + 2 for part in (instructions, style, diff_part))
    changed, budget, missing = contents(files, budget, changed_block)
    context, budget, dropped = contents(deps, budget, context_block)
    parts = [instructions, style] + [block for _, block in context]
    notes = []
    if dropped:
        notes.append(f"# Context dropped for size: {', '.join(dropped)}")
    unreviewed = [
        path for path in missing if f"diff --git a/{path} " not in diff]
    diff_only = [path for path in missing if path not in unreviewed]
    if diff_only:
        notes.append(
            "# Changed files past the budget, reviewed from the diff "
            f"only: {', '.join(diff_only)}")
    if unreviewed:
        notes.append(
            "# Changed files past the budget with no diff left after "
            f"truncation, not reviewed at all: {', '.join(unreviewed)}")
    for note in notes:
        if len(note) + 2 <= budget:
            parts.append(note)
            budget -= len(note) + 2
    parts += [block for _, block in changed]
    parts.append(diff_part)
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
    with open(os.path.join(DIRECTORY, "diff.patch")) as file:
        diff = file.read()
    findings = ask(assemble(files, diff))
    with open(os.path.join(DIRECTORY, "findings.json"), "w") as file:
        json.dump(findings, file)
    print(f"{len(findings.get('findings', []))} findings.")


if __name__ == "__main__":
    main()
