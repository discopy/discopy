# TODO

Cubic's third review round on #610 (run 7edf5232), verbatim:

> P2 (`post.py:69`, x2): When a reused issue has more than 30 comments, the
> script patches an older comment because the comments request accepts only
> GitHub's first page. Fetch and paginate the comments before selecting the
> actual last comment, then update that comment.

> P2 (`codebase-review.yml:28`): When a manual run supplies `inputs.model`
> without `CODEBASE_REVIEW_MODEL`, this check still marks the workflow
> unconfigured and skips the run. Check the effective model override in
> `MODEL` so the documented input can actually replace the variable.

> P2 (`codebase-review.yml:31`): When the bot credentials are missing, the
> configuration check passes and the workflow can spend up to 120 minutes
> reading before token generation fails. Check `DISCOPY_BOT_APP_ID` and
> `DISCOPY_BOT_PRIVATE_KEY` for non-dry runs, or fail early with the
> existing configuration notice.

- [ ] comment pagination: request the single newest comment directly
      (`per_page=1&sort=created&direction=desc`) instead of the first page
- [ ] config check: `MODEL: ${{ inputs.model || vars.CODEBASE_REVIEW_MODEL }}`
- [ ] config check: also require the bot credentials, unless it's a dry run
