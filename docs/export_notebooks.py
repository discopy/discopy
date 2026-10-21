#!/usr/bin/env python
"""Render the marimo notebooks in ``docs/notebooks`` for the documentation.

For every ``docs/notebooks/*.md`` marimo notebook this module:

1. runs it and exports the *computed* result to a self-contained HTML file in
   ``docs/_static/notebooks/<name>.html`` (via ``marimo export html``), and
2. writes a small reStructuredText page ``docs/notebooks/<name>.rst`` that
   embeds that HTML in an ``<iframe>`` so Sphinx picks it up in the toctree.

Both artefacts are generated (they are *not* committed): ``conf.py`` calls
:func:`generate` from a ``builder-inited`` hook, so ``sphinx-build`` renders the
notebooks as part of a normal docs build. If a notebook cannot be executed
(e.g. an optional dependency is missing on Read the Docs) the build still
succeeds -- the page falls back to a link to the notebook source.

It can also be run as a script::

    python docs/export_notebooks.py            # render every notebook
    python docs/export_notebooks.py qnlp        # render a single notebook
    python docs/export_notebooks.py --check     # only check they execute
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"
HTML_DIR = DOCS / "_static" / "notebooks"

IFRAME = """\
.. raw:: html

    <iframe class="marimo-notebook" src="../_static/notebooks/{name}.html"
            title="{title}" loading="lazy"
            style="width: 100%; height: 90vh; border: none;"></iframe>
"""

THEME_STYLE = """\
<style type="text/css">
/* DisCoPy SVGs turn the elements tagged by drawing.backend.DARK_MODE_STYLE
   white under a dark prefers-color-scheme; rendered inline in the notebook,
   they read the browser preference rather than the notebook theme, so these
   higher-specificity rules retag them by marimo's theme class instead. */
.light [id^="dark-stroke"] path { stroke: #000000 !important; }
.light [id^="dark-fill"] g, .light [id^="dark-fill"] use
{ fill: #000000 !important; stroke: #000000 !important; }
.dark [id^="dark-stroke"] path { stroke: #ffffff !important; }
.dark [id^="dark-fill"] g, .dark [id^="dark-fill"] use
{ fill: #ffffff !important; stroke: #ffffff !important; }
</style>"""

FALLBACK = """\
.. note::

    This notebook could not be rendered in this build (a dependency needed to
    execute it may be missing). You can read its source on GitHub:
    `{name}.md <https://github.com/discopy/discopy/blob/main/docs/notebooks/{name}.md>`_.
"""


def title_of(notebook: Path) -> str:
    """The first level-one heading of ``notebook``, outside of code cells.

    Falls back to the ``title:`` field of the YAML front-matter, which marimo
    derives from the name of the file, and then to the name of the file.
    """
    lines = notebook.read_text().splitlines()
    closing = [i for i, line in enumerate(lines) if i and line == "---"]
    end = closing[0] + 1 if lines[:1] == ["---"] and closing else 0
    front, body, fenced = lines[1:end - 1] if end else [], lines[end:], False
    for line in body:
        fenced ^= line.startswith("```")
        if not fenced and line.startswith("# "):
            return line[2:].strip()
    for line in front:
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip()
    return notebook.stem


def export(notebook: Path, *, check: bool) -> None:
    """Run ``notebook`` and export the computed HTML (unless ``check``)."""
    with TemporaryDirectory() as scratch:
        if check:
            output = Path(scratch) / f"{notebook.stem}.html"
        else:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            output = HTML_DIR / f"{notebook.stem}.html"
        subprocess.run(
            [sys.executable, "-m", "marimo", "export", "html", notebook.name,
             "-o", str(output), "-f"],
            cwd=NOTEBOOKS, check=True)
        add_theme_style(output, title_of(notebook))


def add_theme_style(output: Path, title: str) -> None:
    """Insert :data:`THEME_STYLE` and ``title`` in an exported notebook."""
    text = re.sub("<title>.*?</title>",
                  lambda _: f"<title>{escape(title)}</title>",
                  output.read_text(), count=1, flags=re.DOTALL)
    output.write_text(text.replace("</head>", THEME_STYLE + "</head>", 1))


def write_page(notebook: Path, *, rendered: bool) -> None:
    """Write the reStructuredText page for ``notebook``."""
    title = title_of(notebook)
    body = (IFRAME if rendered else FALLBACK).format(
        name=notebook.stem, title=escape(title))
    (NOTEBOOKS / f"{notebook.stem}.rst").write_text(
        f"{title}\n{'=' * len(title)}\n\n" + body)


def generate(*, strict: bool = False) -> None:
    """Render every notebook to HTML and write its page (used by ``conf.py``).

    When ``strict`` is false a notebook that fails to execute falls back to a
    link page instead of aborting the build.
    """
    for notebook in sorted(NOTEBOOKS.glob("*.md")):
        try:
            export(notebook, check=False)
            write_page(notebook, rendered=True)
        except (subprocess.CalledProcessError, OSError):
            if strict:
                raise
            write_page(notebook, rendered=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*",
                        help="notebook stems to render (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="only check the notebooks execute, write nothing")
    args = parser.parse_args()

    stems = set(args.notebooks)
    selected = sorted(nb for nb in NOTEBOOKS.glob("*.md")
                      if not stems or nb.stem in stems)
    if not selected:
        print("no matching notebooks found", file=sys.stderr)
        return 1

    for notebook in selected:
        print(f"rendering {notebook.name} ...", flush=True)
        export(notebook, check=args.check)
        if not args.check:
            write_page(notebook, rendered=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
