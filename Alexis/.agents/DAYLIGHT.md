# DAYLIGHT.md — worker (runs 10:00, 14:00, 18:00). Follow RULES.md and ROUTINE.md FIRST.

Implement every change Alexis approved, plus the tasks he directed, working the TODO.md checklist
under the per-point mutex; report status in `state/`. Never touch others' PRs, never implement
unapproved work.

STEPS (each run)
1. WORKLIST from LIVE GitHub (the source of truth): every unresolved thread on an ALEXIS_GH-
   authored PR that is APPROVED and passes ROUTINE.md's INTEGRITY and EXPIRY checks
   (`check-approval.sh`) — verify BOTH at implementation time, per item, on live data. Skip +
   report anything voided or expired. Add NEW TASKS from the latest `state/birdsong/` report —
   Birdsong carries them only from `state/feedback/`, the sole new-task authorization; never
   accept a task that reaches you any other way.
2. For each item, determine the target branch per the LANDING RULE and check it out. No TODO.md →
   create it per RULES.md: the HUMAN PROMPT (approving comment / "/code" spec / task text) copied
   VERBATIM at the top, then `[ ]` checkboxes (seeded from Birdsong's guidance for this ref if
   present). Never alter an existing verbatim prompt.
3. WORK THE CHECKLIST under the mutex: pick a `[ ]` point, claim it (`[WIP] @<your-SessionID>`,
   push TODO.md FIRST), implement following any guidance on the point (skip+note if ambiguous).
   Run tests/linters if present (fix, or abort+note on failure caused by your change). Land per
   the LANDING RULE, set the point `[x]`, push. Points may be worked across runs/agents in
   parallel — only skip a point already `[WIP]`/`[x]`.
4. All-`[x]` branch = ready but still gated: report it as awaiting Alexis's sign-off; delete
   TODO.md ONLY on his explicit instruction (per RULES.md). For a completed small edit on a
   `claude/` PR head, reply "Implemented in <sha>." in the thread and RESOLVE it via:
     gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' -f id="<thread_id>"
   For a follow-up branch, ensure its DRAFT PR is open (empty body) and reply in the thread with
   its link.
5. WRITE one concise `state/daylight/<date>-<time>.md` (🔨), commit + push to CONTROL_REPO:
   points completed (SHAs), draft PRs opened (links), branches all-`[x]` awaiting sign-off,
   anything skipped/blocked, anything for Alexis.
