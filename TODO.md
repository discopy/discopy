# TODO

Feedback: `codebase-review.yml`'s `read` job failed
([run 32778946870](https://github.com/discopy/discopy/actions/runs/32778946870),
job 97596430926, head `0c2b489e`). The Claude Code sub-agent (`z-ai/glm-4.6`)
ran 102 turns / ~44 min / $17.72 and finished `"subtype": "success"` with
`permission_denials_count: 8`, but left no `.codebase-review/report.md` on
disk — even after the previous fix on this branch (`0c2b489e`, "Tell the
read to use the Write tool, not a Bash heredoc") told it to use `Write`
instead of a `Bash` heredoc for its deliverables. The dry-run summary step
then hard-failed on `cat .codebase-review/report.md`
(`No such file or directory`), so the workflow's own smoke test went red
with no diagnosis of what the agent did instead.

- [x] Make the dry-run summary step degrade gracefully instead of
      hard-failing when the read produces no `report.md`.
- [WIP] @claude-2026-08-25 00:45 Make the prompt write a real `report.md`
      from early in the read and keep it current with further `Write`
      calls, rather than only once at the very end, and have the agent's
      last action be a `Read` of its own `report.md` to confirm it landed
      before the session ends — so running out of turns, or a relapse into
      the old Bash-heredoc habit deep in a long session, leaves a real file
      on disk instead of nothing.
