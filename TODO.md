# Prompt

USER, in response to Claude's diagnosis of the style-review budget crash seen on
https://github.com/discopy/discopy/actions/runs/32744707942/job/97863357670 (PR #532):
`assemble()` in `.github/style-review/review.py` raises `ValueError` when the *changed*
files alone don't fit `BUDGET`, and `style-review.yml`'s "Call the correctness reviewer"
step then fires anyway because it only checks `steps.post.outputs.clean != 'false'`,
which is never `'false'` when the review step crashed and "Post the review" got skipped.

Claude proposed three options and recommended:

> 1. **Graceful degrade (recommended)**: when a changed file doesn't fit, fall back to
> its diff hunk only instead of the full numbered text — exactly what `assemble` already
> does for imported context files (the `dropped`/note path), just extended to the primary
> changed files instead of raising. [...] Separately from which of those we pick, the
> workflow's fallthrough should be fixed either way — a crashed review step shouldn't
> look like "clean" to the correctness-reviewer gate.

USER: "1"

# Work

- [ ] `assemble()` in `.github/style-review/review.py`: stop raising on changed files
      past budget — drop their full-text `Changed` block (like the existing
      context-deps `dropped` path) and note it, keeping their diff hunks (already in
      `diff_part`) as the review's only signal for those files
- [ ] `.github/style-review/style-review.yml`: stop "Call the correctness reviewer"
      from treating a failed "Review the diff" step as clean
- [ ] `CHANGELOG.md` `[Unreleased]` entry
- [ ] `uv run pflake8 discopy` and `uv run coverage run -m pytest` still clean
