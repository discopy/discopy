# TODO

Cubic's second correctness review of #608 (summoned by discopy-bot), quoted
from its summary:

> - [...] allow PR-controlled code or symlinks to access the workflow environment [...]
> - `.github/workflows/style-review.yml` can skip the correctness reviewer when style-review fails, and unrelated label events can cancel an active run [...]
> - `.github/style-review/post.py` can convert malformed or non-integer findings into an apparent clean review while also allowing more than ten findings through [...]
> - `.github/style-review/review.py` can omit changed paths containing whitespace and fail on gateway responses whose prose or code fences contain braces [...]

- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 concurrency
  group keyed by label or action, so an unrelated label cannot cancel a
  running review
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 the token and
  summon steps survive a style-step failure, so a broken gateway cannot
  block the correctness review
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 `post.py`
  normalises int-like `line` values and raises when every reported finding
  is unreadable, instead of reading model garbage as clean
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 `post.py`
  enforces the ten-finding cap, noting the withheld count in the body
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 `review.py`
  reads the file manifest line by line
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 09:08 decline both
  P0s on their threads — same-repo trust boundary, as ruled last round —
  and the parse-defensively fallback, per the fail-loudly ruling
