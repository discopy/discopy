# TODO

> Fix this one https://github.com/discopy/discopy/issues/501

Issue #501 offers three fixes. Asked which one, Alexis picked the second, *keep the mutation but
authenticate with a PAT or a GitHub App installation that can perform it*:

> Add a PAT secret

> i already setup discopy with a bot we're not using anymore i can probably reuse the same

> we removed it here take a look https://github.com/discopy/discopy/commit/d7f67f7f900796244587cac03e771edcc8d3eb98

That bot is a GitHub App: `vars.DOCS_BOT_APP_ID` and `secrets.DOCS_BOT_PRIVATE_KEY`, minted with
`actions/create-github-app-token`, removed from `build.yml` in #470.

- [x] Mint the App token in `no-todo-on-main.yml` so `convertPullRequestToDraft` is reachable
- [x] Fail the job unless the pull request really came back draft, so the gate can never exit 0
      while a TODO file is present on a non-draft pull request
- [x] Drop `pull-requests: write` from a `pull_request_target` workflow
- [ ] Alexis grants the App **Pull requests: write** on this repository — it only needed
      **Contents: write** to push docs baselines, and the mutation stays FORBIDDEN without it
