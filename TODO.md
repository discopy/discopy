# TODO

Cubic's review round on #610 (run d66cbc29), verbatim:

> P0 (`codebase-review.yml:104`): The model can overwrite `post.py`, then the
> workflow executes that modified file with the write-capable bot token. Run
> an immutable copy from `git show HEAD:...` or verify tracked files are
> unchanged before invoking the poster.

> P1 (`codebase-review.yml:67`): Because the agent reads untrusted issue and
> repository text, `Bash(uv run:*)` lets a prompt-injected repro execute
> arbitrary Python and network code with the action's inherited API-key
> environment. Restrict the allowlist to the exact repro commands and invoke
> them through an environment-scrubbed wrapper that excludes the gateway key.

> P2 (`codebase-review.yml:63`): This workflow does not implement the claimed
> OIDC federation: it still authenticates Claude and the GitHub App with
> long-lived secrets. Add the OIDC exchange and remove the long-lived
> credential path, or correct the workflow's security claim and
> documentation.

> P2 (`codebase-review.yml:77`): In a dry run, a missing or empty report is
> rendered as an informational paragraph and the step exits successfully,
> making a failed read look valid. Fail this step when `report.md` is absent
> or empty before printing the summary.

> P2 (`post.py:67`): On a same-day rerun, the existing issue is reused but
> this unconditional POST appends another bugs comment. Fetch and update the
> existing first bot comment, or skip the POST when unchanged, instead of
> creating duplicate comments.

> P2 (`post.py:60`): An open pull request with the matching title can be
> selected as the review target because the issues endpoint also returns
> pull requests. Exclude entries containing `pull_request` before updating
> or commenting.

- [ ] P0: run `post.py` from a git-pinned copy (`git show HEAD:...`), immune
      to anything the read step wrote to the working tree
- [ ] P1: not a code fix — reply that this is the same residual already
      discussed on this PR (the repro-execution channel is the feature, an
      env-scrubbed wrapper isn't available without forking the action), and
      that the P0 fix now bounds its blast radius
- [ ] P2 OIDC: not a code fix — reply that OIDC is a documented *future
      admin option* in the PR description, never claimed as implemented in
      the workflow file itself
- [ ] P2 dry-run false-green: keep the diagnostic message but exit 1 when
      the report is missing, so the step still fails visibly
- [ ] P2 duplicate bugs comment: PATCH the existing comment on a reused
      issue instead of always POSTing a new one
- [ ] P2 PR-as-issue: exclude `pull_request` entries from the title match
