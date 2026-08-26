# TODO

> actually style should apply everywhere not just code, e.g. prompts, workflows, etc
>
> only thing it shouldn't apply to is generated artefacts e.g. svg

Two comments from toumix on [#633](https://github.com/discopy/discopy/pull/633).

Surveyed the repo's tracked file extensions to separate authored content from
generated artefacts: `docs/_static/**` is entirely `.svg`/`.gif`/`.dot`
drawing baselines (177 files, nothing else), `discopy/*.gif` are two more
baselines living outside that directory, `test/drawing/tikz/**` is 15 more
(`.tikz`/`.tikzstyles`), `test/fixtures/**` is 18 files of binary/data test
fixtures (`.pickle`/`.json`), and `uv.lock` is machine-generated (4179
lines). Everything else tracked (`.py`, `.md`, `.rst`, `.yml`/`.yaml`,
`.toml`, `.css`, `.html`, `.bib`, …) is authored.

- [ ] Widen `style-review.yml`'s diff to the whole repo, excluding the
      generated-artefact paths above
- [ ] Generalise `review.py`'s `language()` from a Python/notebook binary
      choice to a real extension → fence-language mapping, since the diff
      can now include `.yml`, `.rst`, `.toml`, etc.
- [ ] Update `prompt.md`'s framing of "changed file" to match
- [ ] Update the `CHANGELOG.md` entry for this PR to describe the final
      (widened) scope rather than the notebook-only first cut
- [ ] Smoke-test `assemble()` against a diff that mixes a `.py`, a `.md`
      and a `.yml` file, and confirm the exclude patterns actually drop a
      historical SVG/gif/pickle/tikz/uv.lock change
