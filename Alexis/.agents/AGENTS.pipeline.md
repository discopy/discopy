# AGENTS.md

Rules for every agent — human-run or autonomous — that writes code in this repo.

1. **Never commit `TODO.md` to the default branch.** It lives only on working branches; a merge
   gate blocks any PR that still contains it.

2. **Never write or edit a PR title or description.** Human-authored only — open PRs as drafts
   with an empty body.

3. **Every new agent branch starts with a fresh `TODO.md`**: the human prompt copied verbatim at
   the top, then the work as `[ ]` checkboxes. Agents may update it but must **never delete it**
   or alter the verbatim prompt.

4. **Claim a checkbox before working it — a per-point mutex.** Set it to `[WIP] @<SessionID>` and
   push *before* any code change; the committed claim is the lock. If the push is rejected
   (non-fast-forward) or the point is already `[WIP]`/`[x]`, take a different point. Parallel
   across points is fine; only the same point is serialized.

5. **An agent talks only to its own human.** Never reply to another user's comment unless the
   human you act for replied first in that thread or marked the comment with a :rocket: — and
   even then, post only a concise description of how the change was executed, resolving the
   thread if appropriate.

6. **`[x]` marks agent-done; deleting `TODO.md` is the human sign-off.** An agent deletes it only
   on explicit human instruction (a PR/issue comment, or a direct instruction in a session) —
   never on its own judgement. Before instructing deletion the human ensures every point is `[x]`
   or filed as an issue; the deletion is what clears the merge gate.
