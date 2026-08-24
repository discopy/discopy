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


def contents(paths, budget):
    """Pair each path with its text while it fits the budget, also
    returning the leftover budget and the paths dropped."""
    kept, dropped = [], []
    for path in paths:
        with open(path) as file:
            text = file.read()
        if len(text) <= budget:
            kept, budget = kept + [(path, text)], budget - len(text)
        else:
            dropped.append(path)
    return kept, budget, dropped


def numbered(text):
    return "\n".join(
        f"{n} {line}" for n, line in enumerate(text.splitlines(), 1))


def assemble(files, diff):
    """The one prompt: instructions, style guide, context, changes, diff."""
    deps = sorted(
        {dep for path in files for dep in imports(path)} - set(files))
    if len(diff) > BUDGET // 2:
        diff = diff[:BUDGET // 2] + "\n[diff truncated for size]"
    changed, budget, missing = contents(files, BUDGET - len(diff))
    context, _, dropped = contents(deps, budget)
    dropped = missing + dropped
    with open(".github/style-review/prompt.md") as file:
        parts = [file.read()]
    with open("STYLE.md") as file:
        parts.append(f"# STYLE.md\n\n{file.read()}")
    parts += [
        f"# Context (not under review): {path}\n\n```python\n{text}```"
        for path, text in context]
    if dropped:
        parts.append(f"# Dropped for size: {', '.join(dropped)}")
    parts += [
        f"# Changed: {path}\n\n```python\n{numbered(text)}\n```"
        for path, text in changed]
    parts.append(f"# Diff\n\n```diff\n{diff}```")
    return "\n\n".join(parts)


def ask(prompt):
    url = os.environ["BASE_URL"].rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": os.environ["MODEL"], "temperature": 0, "max_tokens": 8192,
        "reasoning": {"enabled": False, "exclude": True},
        "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(url, json.dumps(payload).encode(), {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        message = json.load(response)["choices"][0]["message"]
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
