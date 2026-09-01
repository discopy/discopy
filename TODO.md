# TODO

Review feedback from cubic on #696, quoted verbatim:

> P1: When a quoted tally appears before the generated trailing tally, `TALLY_TAIL` consumes both
> blocks because `.*` spans newlines. `scored()` then loses valid verdicts and `post.tallied()`
> deletes the intervening review body; constrain the tally payload to one line instead of using
> DOTALL. (`.github/style-review/history.py:80`)
>
> P1: A stray PR comment can cancel an in-flight review request before the job `if` skips the stray
> run, and an authorized comment can run concurrently with a push review and post a duplicate round
> for the same head. Key eligible review triggers, including `pull_request`, together, and give
> non-review comments unique groups so they cannot cancel real reviews. (Based on your team's
> feedback about unrelated trigger cancellation.) (`.github/workflows/style-review.yml:18`)
>
> P2: When GitHub returns a submitted review from a deleted account, `history()` raises while
> filtering reviews instead of completing the round. Use the same nullable-user handling as
> `thread.author()` so deleted reviewers are ignored safely. (`.github/style-review/history.py:133`)

- [ ] Fix `TALLY_TAIL` to not span newlines in its captured payload, so a quoted tally earlier in
      the body can never absorb the real trailing one.
- [ ] Key the concurrency group so every trigger this job would actually review (a push, or an
      authorized "@discopy review this") shares one group per pull request, and everything else
      gets its own group that can never collide.
- [ ] Guard `history()`'s author check against a `None` `user` (a deleted account), the same way
      `thread.author()` already does.
- [ ] Add regression tests for all three, run `pflake8`/`.github/tests`.
