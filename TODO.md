# TODO

Review round: cubic on `d2d51c0`, 5 findings.

> P2: When artifact download fails for a transient API, permission, or network
> error, this setting lets the Python step return success without posting a
> comment. Handle only the expected no-artifact case while preserving other
> download failures.

> P2: A PR can replace `previous` with any valid-looking SHA before the
> artifact is uploaded, and the privileged job will present that commit as the
> merge base. Validate the merge base against a trusted compare result or keep
> this value outside PR-writable artifact data.

> P2: When `workflow_run.pull_requests` is empty, `mismatch` accepts any PR
> sharing the run's head repository and branch. PR-controlled benchmark code
> can rewrite `metadata.json`, so a same-branch PR with another base can make
> this privileged job comment on the wrong pull request; require a trusted or
> unambiguous PR association instead of falling through.

> P2: When duplicate benchmark comments exist, `ours` returns the oldest marked
> comment and leaves the newest duplicate stale.

> P3: The new changelog entry for `.github/actions/setup` describes the
> Graphviz install as a plain step, but the action retries with `apt-get
> update` after any failed install.

- [ ] 1. Distinguish "no artifact staged" from "the download failed"
- [ ] 2. Validate `previous` against a merge base the privileged job computes
- [ ] 3. Require an unambiguous pull request when the run lists none
- [ ] 4. `ours` takes the newest marked comment
- [ ] 5. P3: reply — the entry does not describe the install mechanics
- [ ] 6. Tests for each, CHANGELOG, green CI
