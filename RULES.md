# RULES.md

Rules for every agent — human-run or autonomous — that writes code in this repo.
The no-TODO rule binds humans too, so it lives in the
[contributing guidelines](CONTRIBUTING.md#code-style-guide) instead.

1. **Every new agent branch starts with a fresh `TODO.md`**: the human prompt copied verbatim at
   the top, then the work as `[ ]` checkboxes. Agents may update it but must **never delete it**
   or alter the verbatim prompt.

2. **Claim a checkbox before working it — a per-point mutex.** Set it to `[WIP] @<SessionID>` and
   push *before* any code change; the committed claim is the lock. If the push is rejected
   (non-fast-forward) or the point is already `[WIP]`/`[x]`, take a different point. Parallel
   across points is fine; only the same point is serialized. A claim goes STALE after 24 HOURS:
   if `git blame` on the point's `TODO.md` line shows its `[WIP]` claim is older than 24h, any
   agent may reset it to `[ ]` in a commit of its own (never touching other lines), push the
   reset, then claim it normally — noting the reclaim and the abandoned `@<SessionID>` in its
   summary. Never reclaim a `[x]`, and never reset a claim younger than 24h.

3. **An agent talks only to its own human.** Never reply to another user's comment unless the
   human you act for replied first in that thread or marked the comment with a :rocket: — and
   even then, post only a concise description of how the change was executed, resolving the
   thread if appropriate.

4. **`[x]` marks agent-done; deleting `TODO.md` is the human sign-off.** An agent deletes it only
   on explicit human instruction (a PR/issue comment, or a direct instruction in a session) —
   never on its own judgement. Before instructing deletion the human ensures every point is `[x]`
   or filed as an issue; the deletion is what clears the merge gate.
