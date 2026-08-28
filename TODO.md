# Round of review feedback from @nightmare6728 on #676

> **The defect** is inline on `post.py`: the early return when the head has
> moved skips `record(clean=…)`, and the workflow's handover step treats the
> unset output as clean, so a round that stood down still calls the correctness
> reviewer — permanently, because the `called` guard then stops the round that
> actually reviews the new head from doing it. One line.

> **The gap** is what this pull request does *not* absorb from #671. […] `ask()`
> still wraps a bare `urlopen`, and the `http.client.IncompleteRead` that killed
> the review run on #661 four minutes into a chunked response would kill it
> again. […] catch `urllib.error.URLError` alongside `IncompleteRead` and
> `TimeoutError`, since a reset connection is the same failure; and with
> `timeout=600` and no backoff the worst case is three ten-minute waits, so a
> shorter timeout or a cap is worth having.

> 1. The size-drop notes sit inside the append-only prefix — inline on
>    `review.py`.
> 2. […] Worth naming the paths in the body line rather than only the count, so
>    a dropped remark is at least locatable in the job log.

- [x] `record(clean=False)` before the moved-head return, with a test entering
      that branch
- [x] move the size-drop notes out of the append-only prefix, after the
      discussion and before the changed files they describe
- [x] port #671's retry as `complete()`, catching `URLError` but never
      `HTTPError`, with the attempts capped inside the job's timeout, and tests
- [x] answer point 2 on the thread: the dropped paths are already printed to
      the job log, and the body deliberately lists no finding
