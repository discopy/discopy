# TODO

Human prompt, the issue this branch fixes, quoted verbatim:

> ## discopy#695 — Style-review machinery: nine findings from cubic's review of the stack
>
> Cubic reviewed [#665](https://github.com/discopy/discopy/pull/665) against a stale merge-base, so
> its diff included `main`'s style-review memory machinery (landed via #676). Nine of its ten
> findings target that code, not the zx slice — collected here so a PR on `main` can fix them,
> source: [the review](https://github.com/discopy/discopy/pull/665#pullrequestreview-5059976415).
>
> **Trust / forgery** (`.github/style-review/history.py`):
> - `history()` accepts any comment starting with the public `MARKER` carrying valid JSON as a bot
>   round — nothing authenticates the author, so review history and verdicts can be forged (line
>   ~56).
> - `scored` parses a quoted tally-looking marker as generated and imports its verdicts; parse only
>   the exact separator `post.tallied` emits (line ~78).
> - `post.tallied()` likewise deletes quoted marker text and everything after it when a body quotes
>   the tally after a blank line (post.py ~187).
>
> **Robustness**:
> - `remarklike` accepts a record with a non-string `comment`; `review.past_block`/`literal()` then
>   crash on `.strip()` (history.py ~48, review.py ~176) — validate field types at the boundary.
> - `review.complete()` retries `IncompleteRead`/`URLError`/`TimeoutError` but not
>   `ConnectionError`/`ConnectionResetError` mid-body (review.py ~274).
>
> **Correctness**:
> - `post.py`'s commentable-line scan lets inter-file metadata rows (`diff --git`, `index`, `new
>   file mode`) fall through to the counter, inflating `lines[path]` past the file's end — an
>   off-diff finding at a coincidental number posts inline, 422s the review, and demotes true
>   inline remarks to the body (post.py ~60; cubic reproduced it).
> - `main()` records `clean=true` when there are no findings even while coverage lists uncovered
>   files — a partial review hands over to the correctness reviewer as clean (post.py ~257).
> - `style-review.yml`'s concurrency group keys `issue_comment: created` and
>   `pull_request: synchronize` together, so a manual request cancels an active push review and
>   vice versa — key by `github.event_name` too (yml ~15).

- [ ] Authenticate `history()`'s reviews by their poster, so only the discopy-bot's own rounds are
      read back as history — a comment or review from anyone else starting with `MARKER` is
      somebody quoting it, not a round.
- [ ] Validate `remarklike`'s field types (`path`/`comment` strings, `line` an int), so a malformed
      record cannot reach `review.past_block`/`literal()` and crash on `.strip()`.
- [ ] Anchor `history.scored()` to the exact trailing shape `post.tallied()` emits, not any quoted
      marker text found anywhere in the body.
- [ ] Anchor `post.tallied()` the same way, so a remark that quotes the marker after a blank line
      is left alone rather than truncating the body from there.
- [ ] Fix `post.commentable_lines()` to reset the current path at each `diff --git` boundary, so
      inter-file metadata rows never inflate the previous file's line count.
- [ ] Fix `post.main()` to record `clean=false` when the round could not read every changed file,
      even when it made no finding.
- [ ] Catch `ConnectionError` in `review.complete()`'s retry, alongside `IncompleteRead`/
      `URLError`/`TimeoutError`.
- [ ] Key `style-review.yml`'s concurrency group by `github.event_name` too, so an unrelated PR
      comment cannot cancel an in-flight push's review round.
- [ ] Add an entry to `CHANGELOG.md`'s `[Unreleased]` section.
- [ ] Run `pflake8` and the `.github/tests` suite.
