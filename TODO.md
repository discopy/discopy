# TODO

Round 1 of review feedback, from cubic (the correctness reviewer):

> P2: When a finding comment contains a newline, this interpolation breaks the
> numbered history entry into multiple prompt lines. Escape or quote the remark
> text before inserting it, as `quoted()` already does for replies.
> (`.github/style-review/review.py:150`)

> P2: After enough review rounds, the unbounded history block can consume the
> entire prompt budget, so `assemble()` drops every changed file and raises
> before calling the gateway. Cap or truncate the history while reserving space
> for the current changed files. (`.github/style-review/review.py:170`)

> P2: When a finding contains `<!-- style-review-tally -->`, `retally()`
> truncates the first review at that text before appending the tally. Strip only
> the generated trailing tally. (`.github/style-review/post.py:95`)

> P2: A review body that starts with `<!-- style-review ` but contains malformed
> JSON crashes `recorded()` and aborts the whole history step. Catch parse errors
> and treat invalid markers as non-style-review comments so one bad body cannot
> block later rounds. (`.github/style-review/history.py:34`)

- [x] A past remark goes into the prompt as a literal, like a reply, so a
      newline in it cannot break the numbered listing
- [x] The changed files are budgeted before the past remarks, which are dropped
      whole when they do not fit
- [x] `retally` strips only a trailing tally, not a marker quoted in a remark
- [x] `recorded` reads an unparsable marker as somebody else's review
