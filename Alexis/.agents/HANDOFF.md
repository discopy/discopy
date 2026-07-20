# Hand-off: bootstrap the routine pipeline

Paste this into a fresh Claude Code session with write access to `rel-int/wiki` and
`discopy/discopy`. It sets up the repo side of a four-run/day autonomous pipeline whose phases are
named after the board game Root's turn structure: **Birdsong, Daylight, Evening**. Creating the
routines themselves is the manual checklist at the end — a session can't provision routines.

Accompanying files (commit VERBATIM):
- `AGENTS.md` → the ROOT of `rel-int/wiki` AND of `discopy/discopy` (identical copy in both).
  Short and public-safe by design; sitting at the root, every Claude Code session in either repo
  — yours and the routines — auto-loads it, so the TODO.md/mutex protocol binds all of them.
- `Alexis/.agents/{ROUTINE,EVENING,BIRDSONG,DAYLIGHT}.md` → `rel-int/wiki` only.

## 1. Fill in config
In `Alexis/.agents/ROUTINE.md` complete `## Config`: `ALEXIS_GH`, `ALEXIS_SLACK_ID`,
`SLACK_CHANNEL`. Verify each repo's default branch (DisCoPy may be `master`).

## 2. Commit the files; protect the config with CODEOWNERS
Two pitfalls: CODEOWNERS does nothing unless branch protection enables "Require review from Code
Owners", and a sole owner who authors a PR cannot approve it — so list a second trusted owner
(e.g. Giovanni) alongside Alexis. Open these as normal PRs:

    # rel-int/wiki/.github/CODEOWNERS
    /AGENTS.md                  @toumix @gio-defelice
    /Alexis/                    @toumix @gio-defelice
    /.github/workflows/         @toumix @gio-defelice
    # discopy/.github/CODEOWNERS
    /AGENTS.md                  @toumix @gio-defelice
    /.github/workflows/         @toumix @gio-defelice

## 3. Merge gate on BOTH repos (enforces AGENTS.md rule 1)
`TODO.md` must never reach the default branch. Add this workflow to each repo, then require its
check:

    # .github/workflows/no-todo-on-main.yml
    name: no-todo-on-main
    on:
      pull_request:
        branches: [main]        # "master" for DisCoPy if that's its default
    jobs:
      guard:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Block TODO.md on the default branch
            run: |
              if git ls-files | grep -qiE '(^|/)todo\.md$'; then
                echo "::error::A TODO.md is present — a human must review it and delete it before merge."
                exit 1
              fi

Require it — settings differ per repo to avoid deadlocks. WIKI (routine PRs are authored as
Alexis, who can't self-approve — so no review requirement, just the check):

    gh api -X PUT "repos/rel-int/wiki/branches/main/protection" \
      -H "Accept: application/vnd.github+json" \
      -f "required_status_checks[strict]=true" \
      -f "required_status_checks[checks][][context]=guard" \
      -F "enforce_admins=true" \
      -F "required_pull_request_reviews=null" \
      -F "restrictions=null"

DISCOPY (a second maintainer exists — require review + code-owner review):

    gh api -X PUT "repos/discopy/discopy/branches/{branch}/protection" \
      -H "Accept: application/vnd.github+json" \
      -f "required_status_checks[strict]=true" \
      -f "required_status_checks[checks][][context]=guard" \
      -F "enforce_admins=true" \
      -F "required_pull_request_reviews[required_approving_review_count]=1" \
      -F "required_pull_request_reviews[require_code_owner_reviews]=true" \
      -F "restrictions=null"

(`enforce_admins=true` binds even an agent acting with Alexis's admin rights — the guarantee that
only a human-instructed TODO.md deletion can land a branch.)

## 3b. Protect the gate itself (both repos)
A `pull_request` workflow runs the version ON THE PR BRANCH, so an agent could edit
`no-todo-on-main.yml` to always pass. Two layers:
  1. The routines' fine-grained PAT (manual checklist) must NOT have the "Workflows" permission —
     GitHub then rejects any push from it touching `.github/workflows/`.
  2. A repo RULESET on each repo (Rules → Rulesets → push/branch ruleset, all branches, "restrict
     file paths" = `.github/workflows/**`, bypass = Alexis only) — also covers interactive
     sessions, which run with Alexis's full credentials rather than the PAT.

## 4. Report back
Summarize the PRs opened, the merge-gate workflows, and the required-check status. Then STOP —
don't create routines or push `claude/` work.

---

## Manual checklist for Alexis (claude.ai — a session can't do this)

Create THREE routines. Each routine's Instructions field is just a pointer:

> Follow the repo-root `AGENTS.md`, then `wiki/Alexis/.agents/ROUTINE.md`, then
> `wiki/Alexis/.agents/<PHASE>.md`, exactly. Do nothing they don't instruct.

| Routine   | Schedule (your TZ)   | phase file    |
|-----------|----------------------|---------------|
| Evening   | daily 02:00          | EVENING.md    |
| Birdsong  | daily 09:50          | BIRDSONG.md   |
| Daylight  | `0 10,14,18 * * *`   | DAYLIGHT.md   |

Per routine: add both repos; Slack connector; `GH_TOKEN` env (fine-grained PAT, both repos, PR +
Contents read/write, NO "Workflows" permission — see 3b); allow `api.github.com`; leave branch
pushes RESTRICTED. Make `SLACK_CHANNEL` private with only you in it. Five scheduled runs/day sits
at the Pro cap — comfortable on Max/Team/Enterprise.

## The loop, once live
Evening (02:00) posts points + a summary (incl. expired/voided approvals to re-🚀) → you reply on
Slack and 🚀 changes on GitHub (approvals expire after 7 days; an edited comment voids its
approval) → Birdsong (09:50) shapes each approved branch's `TODO.md` into a plan-aligned checklist
with guidance → Daylight (10/14/18) claims each `[ ]` point via the `[WIP] @SessionID` mutex,
implements it, marks `[x]` — never deleting `TODO.md` → when a branch is all `[x]`, **you** review
it, confirm every point is done or filed as an issue, and delete `TODO.md` — the only thing that
clears the merge gate → Evening folds the day's `🔨` summaries back in that night.
