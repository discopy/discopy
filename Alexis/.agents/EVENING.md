# EVENING.md — planner + reporter (runs 02:00). Follow RULES.md and ROUTINE.md FIRST.

Read the repo state and the Slack channel (shared memory), think hard about the day ahead, and
post discussion points + an executive summary for Alexis to answer in the morning. Report-only —
you never modify code.

STEPS
1. READ MEMORY: SLACK_CHANNEL since the previous Evening — Alexis's untagged feedback and the
   "🔨 DAYLIGHT" summaries.
2. SCAN GITHUB (all REPOS): open issues/PRs (author, review decision, requested reviewers, CI,
   base/head, review threads + reactions); dependency graph; blockers. Queues:
     - needs_alexis: open PRs where a review/decision waits on ALEXIS_GH.
     - awaiting_approval: unresolved threads on ALEXIS_GH-authored PRs with a pending change not
       yet approved (no :rocket:, no "/code"). Keep direct comment URLs.
     - expired_or_voided: approvals failing ROUTINE.md's EXPIRY or INTEGRITY checks — with links
       so Alexis can re-approve if still wanted.
     - awaiting_signoff: `claude/` branches all-`[x]` — waiting on Alexis to delete TODO.md.
3. THINK HARD: improvements, risks, priorities, longer-term directions. Curate to a handful.
4. POST:
   - One "🌙 POINT [<date> · P<n>]" per point, with links and a clear question. ≤ ~5 points.
   - One "🌙 SUMMARY [<date>]": what Daylight did (from 🔨 summaries) + GitHub changes; "Awaiting
     your :rocket:" (comment links); "Expired / voided approvals — re-approve if still wanted"
     (links); "Awaiting your sign-off" (branches ready — just delete TODO.md); "Needs your
     decision" (links); a nudge to reply.
