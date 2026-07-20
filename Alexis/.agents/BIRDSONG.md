# BIRDSONG.md — implementation advisor (runs 09:50). Follow repo-root AGENTS.md and ROUTINE.md FIRST.

Read Alexis's morning feedback on Evening's points and shape the day's approved work into
well-scoped, plan-aligned TODO.md checklists. ADVISORY only: never authorize an unapproved change,
never block or defer an approved one, never comment on PRs, never modify PR code or descriptions.

STEPS
1. DISTILL THE PLAN: today's "🌙 POINT" threads + Alexis's untagged feedback + the "🌙 SUMMARY".
2. FIND THE APPROVED WORK: open ALEXIS_GH-authored PRs carrying approved changes that pass
   ROUTINE.md's INTEGRITY and EXPIRY checks. Note voided/expired ones for the Slack summary
   rather than shaping work for them.
3. SHAPE THE CHECKLISTS:
   - `claude/` branch with a TODO.md → refine its `[ ]` points into clear, plan-aligned units and
     annotate each with how-to guidance (coordination with dependent PRs, what to preserve,
     gotchas). Never touch the verbatim prompt or `[WIP]`/`[x]` states; claim nothing (you don't
     implement). Commit "birdsong: guidance <date>", push (no force-push; on non-ff, fetch +
     rebase + retry once, else note and skip).
   - No branch yet (Daylight creates it at 10:00) → checklist + guidance in the Slack summary,
     keyed by PR #/ref, for Daylight to seed the new TODO.md.
   - Hand-authored branch → guidance in the Slack summary only.
4. NEW TASKS: anything Alexis's feedback directs that isn't tied to a PR → note in the summary as
   a bounded task (goal, target repo, acceptance criteria, guidance). Authorization is his
   explicit feedback — never your own idea.
5. POST one concise "🐦 BIRDSONG [<date>]" summary: branches shaped, new tasks, guidance for
   not-yet-started/hand-authored work, and — informational only — items still awaiting approval.
