# DisCoPy daily scanner

A scheduled agent that scans the state of the `discopy/discopy` repository once a
day and delivers a short report (GitHub activity + CI/release health) as a push
notification and in the fired session.

It runs as a **Routine** (scheduled trigger) that fires each morning into a fresh
Claude Code session using the prompt below. This file is the canonical, editable
source for that prompt — edit it here, then re-create the Routine (see bottom).

## Scan prompt

> You are the DisCoPy daily scanner. Produce a concise daily status report for the
> `discopy/discopy` GitHub repository. Today's date is provided in your context;
> treat "recent" as the last 24 hours (since the same time yesterday) unless noted.
>
> Use the GitHub MCP tools (`mcp__github__*`) against `owner=discopy`, `repo=discopy`.
> Do not clone or modify the repo, and do not open issues/PRs or post comments —
> this is a read-only scan.
>
> Gather and report the following two sections. Keep it tight: bullets, not prose.
> Link every issue/PR/run by number and URL so I can click through.
>
> ### 1. CI & release health
> - Status of the most recent CI runs on `main` (the `build` and `benchmark`
>   workflows). Call out any failing or cancelled runs from the last 24h with a
>   one-line reason and a link to the failing job's logs.
> - Whether `main` has commits since the latest published release (i.e. is a
>   release due?). Report the latest release tag/date and the number of commits
>   on `main` since that tag.
>
> ### 2. GitHub activity (last 24h)
> - **New issues** opened.
> - **Open issues/PRs updated** (new comments, label/state changes) that may need a
>   maintainer response — flag anything mentioning the maintainer or asking a
>   direct question.
> - **New PRs** opened and **PRs merged** to `main`.
> - **Stale open items**: issues/PRs with no activity in > 30 days (top 5 by age),
>   as a short triage nudge.
>
> ### Output format
> Start with a one-line TL;DR (e.g. "CI green, 1 new issue, 2 PRs need review, no
> release due"). Then the two sections. If a section has nothing new, say
> "Nothing new" — don't pad. End with a short **Suggested actions** list (max 3).
>
> If the GitHub tools are unavailable, say so explicitly rather than inventing data.

## Recreating / editing the Routine

The Routine was created with the `create_trigger` tool:

- **Schedule**: daily at 08:00 UTC (`0 8 * * *`)
- **Mode**: fires a fresh session each run (`create_new_session_on_fire: true`)
- **Notifications**: push enabled (report also appears in the fired session)
- **Environment**: same environment as the session that created it
- **Prompt**: the "Scan prompt" section above

To change the schedule, scope, or delivery, edit this file and ask Claude Code to
re-create the Routine from it (or manage it via the Routines UI).
