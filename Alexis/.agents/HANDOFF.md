# Hand-off: bootstrap the routine pipeline

Paste this into a fresh Claude Code session with write access to CONTROL_REPO (Alexis's personal
repo, the one for his website) and `discopy/discopy`. It sets up the repo side of a four-run/day
autonomous pipeline whose phases are named after the board game Root's turn structure:
**Birdsong, Daylight, Evening**. Creating the routines themselves is the manual checklist at the
end.

Accompanying files (commit VERBATIM):
- `AGENTS.md` → the ROOT of CONTROL_REPO AND of `discopy/discopy` (identical copy in both).
  Short and public-safe by design; sitting at the root, every Claude Code session in either repo
  — yours and the routines — auto-loads it, so the TODO.md/mutex protocol binds all of them.
- `Alexis/.agents/{ROUTINE,EVENING,BIRDSONG,DAYLIGHT}.md` + `check-approval.sh` → CONTROL_REPO
  only. The routines also write a `state/` folder there (see ROUTINE.md) — git is the memory
  bus, the bridge chat is the feedback UI; there is no Slack in the loop.

## 1. Fill in config
In `Alexis/.agents/ROUTINE.md` complete `## Config`: `ALEXIS_GH`, `CONTROL_REPO`. Verify each
repo's default branch (DisCoPy may be `master`). If CONTROL_REPO is public, treat the prompts as
published (the approval scheme is identity-based and survives that) — but keep any genuinely
private values in the routines' env, not in these files.

## 2. Commit the files; keep the control plane out of the website build
If CONTROL_REPO builds a static site, exclude `Alexis/` and `state/` in the generator's config so
the agent files and memory never appear as pages. For discopy, protect the shared config with
CODEOWNERS — it does nothing unless branch protection enables "Require review from Code Owners",
and a sole owner who authors a PR cannot approve it, so list a second trusted owner (e.g.
Giovanni) alongside Alexis:

    # discopy/.github/CODEOWNERS
    /AGENTS.md                  @toumix @gio-defelice
    /.github/workflows/         @toumix @gio-defelice

## 3. Merge gate on discopy (enforces AGENTS.md rule 1)
`TODO.md` must never reach the default branch. Add this workflow, then require its check:

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
only a human-instructed TODO.md deletion can land a branch. CONTROL_REPO needs no merge gate:
routines never create TODO.md there — they push only `state/` files.)

## 3b. Protect the gate and the rules themselves
A `pull_request` workflow runs the version ON THE PR BRANCH, so an agent could edit
`no-todo-on-main.yml` to always pass. Layers:
  1. The routines' fine-grained PAT (manual checklist) must NOT have the "Workflows" permission —
     GitHub then rejects any push from it touching `.github/workflows/`.
  2. A repo RULESET on discopy (Rules → Rulesets → push/branch ruleset, all branches, "restrict
     file paths" = `.github/workflows/**`, bypass = Alexis only) — also covers interactive
     sessions, which run with Alexis's full credentials rather than the PAT.
  3. On CONTROL_REPO there is no per-path PAT permission for `Alexis/.agents/`, so a routine
     sharing Alexis's identity could technically edit its own rules. ROUTINE.md forbids it in
     prose; to make it mechanical, add a push RULESET restricting `Alexis/.agents/**` with NO
     bypass — then even Alexis changes the rules via PR, which is the right ceremony anyway. The
     repo's tiny history is the audit trail either way.

## 4. Report back
Summarize the PRs opened, the merge-gate workflow, and the required-check status. Then STOP —
don't create routines or push `claude/` work.

---

## Manual checklist for Alexis

FIRST, the bridge: open a fresh Claude session in an environment with both repos as sources —
this session IS the bridge (keep it pinned; your replies in it are the feedback channel). From
inside it, ask it to create the Evening routine bound to itself (a self-bound / persistent-
session routine firing daily at 02:00 with the pointer below). Rotate the bridge roughly monthly:
since all durable state lives in `state/`, rotation is free — open a new session, re-bind
Evening, archive the old one.

THEN create Birdsong and Daylight as fresh-session routines (claude.ai or any session). Each
routine's Instructions field is just a pointer:

> Follow the repo-root `AGENTS.md`, then `<CONTROL_REPO>/Alexis/.agents/ROUTINE.md`, then
> `<CONTROL_REPO>/Alexis/.agents/<PHASE>.md`, exactly. Do nothing they don't instruct.

| Routine   | Schedule (your TZ)   | phase file    | session                  |
|-----------|----------------------|---------------|--------------------------|
| Evening   | daily 02:00          | EVENING.md    | bound to the bridge      |
| Birdsong  | daily 09:50          | BIRDSONG.md   | fresh per fire           |
| Daylight  | `0 10,14,18 * * *`   | DAYLIGHT.md   | fresh per fire           |

Per routine: add both repos; `GH_TOKEN` env (fine-grained PAT, both repos, PR + Contents
read/write, NO "Workflows" permission — see 3b); allow `api.github.com`; leave branch pushes
RESTRICTED. Five scheduled runs/day sits at the Pro cap — comfortable on Max/Team/Enterprise.

## The loop, once live
Evening (02:00) posts points + a summary into the bridge chat and mirrors them to
`state/evening/` (incl. expired/voided approvals to re-🚀) → you reply in the bridge whenever —
Evening quotes your feedback VERBATIM into `state/feedback/` — and 🚀 changes on GitHub
(approvals expire after 7 days; an edited comment voids its approval; `check-approval.sh` is the
arbiter) → Birdsong (09:50) reads `state/` and shapes each approved branch's `TODO.md` into a
plan-aligned checklist with guidance, written to `state/birdsong/` → Daylight (10/14/18) claims
each `[ ]` point via the `[WIP] @SessionID` mutex, implements it, marks `[x]` — never deleting
`TODO.md` — and reports to `state/daylight/` → when a branch is all `[x]`, **you** review it,
confirm every point is done or filed as an issue, and delete `TODO.md` — the only thing that
clears the merge gate → Evening folds the day's 🔨 reports back in that night.
