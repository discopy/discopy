# RULES.md

Rules for every agent — human-run or autonomous — that works on a branch or pull request in this repo.

1. **Agent branches start with a fresh `TODO.md`**: the human prompt copied verbatim at the top, then the work as `[ ]` checkboxes. Agents may update it but must never alter the verbatim prompt without explicit human instruction (a PR/issue comment, or a direct instruction in a session). Once every point is `[x]` or filed as an issue, the agent deletes `TODO.md` itself: the deletion clears the merge gate, takes the pull request out of draft and hands it to the reviewers, starting with the style review.

2. **Checkboxes act as mutex.** Commit a one line change from `[ ]` to `[WIP] @<SessionID>-<yyyy-MM-dd HH:mm>` and push *before* you start doing any work; the commit is the lock. If the push is rejected (non-fast-forward) or the point is already `[WIP]`/`[x]`, you can start working on a different point in parallel if it doesn't conflict with ongoing work. Claims go stale after 12 hours: just reclaim it by pushing a new one-line commit.

3. **Published PR history is append-only.** When updating a pull request from its target branch, check out the PR branch, fetch, and merge the target branch into it; never rebase the PR branch onto the target. Push normally—never force-push. If another worker makes the push non-fast-forward, fetch and merge the remote PR branch into the local PR branch, rerun the relevant checks, and retry the normal push. Preserve published commit IDs so review state and other workers’ checkouts remain valid.

4. **Keep it concise, only talk when prompted.** Never reply to another user's comment unless the human you act for replied first or marked the comment with a :rocket: emoji. Post only a concise description of how the change was executed, resolving the thread if appropriate.
