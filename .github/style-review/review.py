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
    """Pair each path with its rendered ``block`` while it fits the
    budget, also returning the leftover budget and the paths dropped."""
    kept, dropped = [], []
    for path in paths:
        with open(path) as file:
            text = block(path, file.read())
        if len(text) <= budget:
            kept, budget = kept + [(path, text)], budget - len(text)
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
    Every part is budgeted as assembled, not as raw file text, so the
    request sent to the gateway never exceeds ``BUDGET``."""
    deps = sorted(
        {dep for path in files for dep in imports(path)} - set(files))
    if len(diff) > BUDGET // 2:
        diff = diff[:BUDGET // 2] + "\n[diff truncated for size]"
    diff_part = f"# Diff\n\n```diff\n{diff}```"
    with open(".github/style-review/prompt.md") as file:
        instructions = file.read()
    with open("STYLE.md") as file:
        style = f"# STYLE.md\n\n{file.read()}"
    budget = BUDGET - len(instructions) - len(style) - len(diff_part)
    changed, budget, missing = contents(files, budget, changed_block)
    if missing:
        raise ValueError(f"changed files past the budget: {missing}")
    context, _, dropped = contents(deps, budget, context_block)
    parts = [instructions, style] + [block for _, block in context]
    if dropped:
        parts.append(f"# Context dropped for size: {', '.join(dropped)}")
    parts += [block for _, block in changed]
    parts.append(diff_part)
    return "\n\n".join(parts)


def ask(prompt, disable_reasoning=True):
    """One chat completion. Retries once without ``reasoning`` when the
    gateway rejects it as disabled, for models that mandate it."""
    url = os.environ["BASE_URL"].rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": os.environ["MODEL"], "temperature": 0, "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}]}
    if disable_reasoning:
        payload["reasoning"] = {"enabled": False, "exclude": True}
    request = urllib.request.Request(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            message = json.load(response)["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        print(f"gateway error {error.code}: {body}", file=sys.stderr)
        if disable_reasoning and "reasoning" in body.lower():
            return ask(prompt, disable_reasoning=False)
        raise
    answer = message.get("content") or message.get("reasoning") or ""
    if "{" not in answer or "}" not in answer:
        raise ValueError(f"no JSON in the gateway answer: {answer[:200]!r}")
    return json.loads(answer[answer.index("{"):answer.rindex("}") + 1])


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
