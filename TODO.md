# TODO

> Fix this one https://github.com/discopy/discopy/issues/501

Issue #501 offers three fixes. Asked which one, Alexis picked the second, *keep the mutation but
authenticate with a PAT or a GitHub App installation that can perform it*:

> Add a PAT secret

> i already setup discopy with a bot we're not using anymore i can probably reuse the same

> we removed it here take a look https://github.com/discopy/discopy/commit/d7f67f7f900796244587cac03e771edcc8d3eb98

That bot is a GitHub App, minted with `actions/create-github-app-token`, removed from `build.yml`
in #470. It is being renamed from the docs bot it used to be:

> i updated the app's permissions
> help me rename it to remove "docs" from its title too, it should be just discopy-bot

- [x] Mint the App token in `no-todo-on-main.yml` so `convertPullRequestToDraft` is reachable
- [x] Fail the job unless the pull request really came back draft, so the gate can never exit 0
      while a TODO file is present on a non-draft pull request
- [x] Drop `pull-requests: write` from a `pull_request_target` workflow
- [x] Alexis grants the App **Pull requests: write** on this repository
- [x] Read the App id and private key from `DISCOPY_BOT_APP_ID` / `DISCOPY_BOT_PRIVATE_KEY`
- [x] Alexis renames the App to `discopy-bot` in its settings, and re-creates the repository
      variable and secret under the new names — neither rename can be done from here

## Daylight follow-up

> for the draft gate I want the if and only if: make the PR ready automatically when the TODO is deleted

> approve gate design

> implement it and push to the existing branch

- [x] Make TODO inspection fail closed and the gate bidirectional for gate-managed pull requests
- [ ] Make `guard` a required status check on `main`
