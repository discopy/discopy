# Review round — cubic-dev-ai on #617

> P2 (confidence 9): When the diff exceeds half the budget, this fallback can omit a
> changed file's full text after truncation removes its hunk, losing the change rather
> than only its context. Preserve each changed file's hunk before dropping full context.
> (`.github/style-review/review.py:88`)

> P2 (confidence 6): When the diff is larger than BUDGET//2, diff_part is truncated and
> the tail of diff.patch is lost; the same big file whose full text didn't fit can have
> all of its hunks inside that truncated tail. The new note then tells the model the
> file was 'reviewed from the diff only' while no diff for it is present [...] The
> diff-only fallback is only sound when the file's hunks actually survived truncation;
> the note should reflect that (or the note should be suppressed when the diff was
> truncated and the file isn't in the surviving diff). (`.github/style-review/review.py:108`)

> P2 (confidence 7): The new `steps.review.outcome != 'failure'` guard suppresses the
> correctness reviewer [...] whenever the style-review step fails for any reason —
> gateway HTTPError, ask() raising, timeout, or any future bug — silently dropping the
> correctness sign-off for the PR. It also doesn't stop the false-'clean' case it
> targets [...] Consider keeping the correctness summons on a failed review [...] and
> instead distinguishing a real crash from a clean result in the post step, rather than
> gating the correctness reviewer on the style tool's health.
> (`.github/workflows/style-review.yml:93`)

# Work

- [ ] `review.py`: make the "reviewed from the diff only" note honest — check whether a
      changed file's hunk actually survived the (possibly truncated) diff text, and
      split the note into "reviewed from the diff only" vs. "no diff left after
      truncation, not reviewed at all"
- [ ] `style-review.yml`: revert the `steps.review.outcome != 'failure'` guard (it
      blocks correctness review on unrelated crashes without correctly targeting the
      scenario it meant to); instead have the correctness-reviewer step's own comment
      body say the style review crashed when `steps.review.outcome == 'failure'`
- [ ] `pflake8` clean
