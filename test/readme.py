"""Extract the Python code blocks from README.md and check they run."""

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

README = Path(__file__).parent.parent / "README.md"

CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_readme(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # to_gif saves a gif in the working directory
    matplotlib.use("Agg")
    text = README.read_text(encoding="utf-8")
    blocks = list(CODE_BLOCK.finditer(text))
    assert blocks, f"no Python code blocks found in {README}"
    namespace = {}  # blocks share state, e.g. `sentence` is reused throughout
    for match in blocks:
        # Pad with newlines so tracebacks point at the actual README lines.
        source = "\n" * text.count("\n", 0, match.start(1)) + match.group(1)
        try:
            code = compile(source, str(README), "exec")
        except SyntaxError:
            continue  # pseudo-code block, e.g. `f @ g = ... != ...`
        exec(code, namespace)
        plt.close("all")
