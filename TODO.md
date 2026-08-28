# TODO

Round 1 of correctness review, from cubic: 16 findings across 14 files, which
are six defects.

> P1: After a remark is accepted or declined, the next history refresh discards
> that verdict, allowing a later round to overwrite the tally with `still open`.
> Persist each verdict with the remark record or merge prior verdicts before
> generating the new tally. (`history.py:81`, `review.py:172`, `review.py:287`,
> `post.py:198` twice)

> P2: This rule makes resolved remarks regress to `open`. [...] A remark that was
> fixed in an earlier round is forced back to `open` the moment that file stops
> changing, because the model can no longer see it. (`prompt.md:63`)

> P2: A valid but non-list marker payload crashes `history()` while numbering
> remarks. (`history.py:45`, `history.py:77`)

> P2: When GitHub returns a pending review carrying this marker, `history()`
> counts it as a posted round. (`history.py:73`)

> P2: `anchor()` gives distinct threads on the same path and line the same
> retention key (`thread.py:23`); protected entries can exceed `BUDGET`
> (`thread.py:72`); the omission note is added after the budget is enforced
> (`thread.py:79`)

> P2: When `history.py` fails, GitHub skips `Review the diff` [...] so it sends a
> normal correctness summon instead of reporting the style failure
> (`style-review.yml:125`); the crash comment is not recognised by the `called`
> check (`style-review.yml:184`)

> P2: This says context lines inside a diff are rejected, but `post.py` treats
> every line in each hunk as commentable (`prompt.md:10`)

> P2: Only an inline-comment validation rejection should trigger the body-only
> retry (`post.py:145`)

- [ ] A verdict once decisive survives a later round that forgets it: the tally
      comment carries the verdicts, `post.py` merges rather than recomputes
- [ ] The prompt stops forcing `open` on a remark whose file it cannot see, and
      asks for silence on it instead
- [ ] `recorded` reads a marker of the wrong shape as somebody else's review
- [ ] A review nobody submitted is not a round
- [ ] `thread.render` keys its protection by the thread, counts its own note,
      and bounds what it protects
- [ ] The handoff tells the correctness reviewer when the style review never
      ran, and says it in a comment the `called` check recognises
- [ ] `commentable_lines` gives the lines the diff adds, which is what the
      prompt promises
- [ ] Only a 422 falls back to the body
