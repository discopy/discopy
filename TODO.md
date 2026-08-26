# TODO

> let's fix this weird workflow bug https://github.com/discopy/discopy/issues/640

- [ ] read the live `isDraft` instead of the `github.event.pull_request.draft` snapshot
- [ ] serialise the guard per pull request so a live read cannot race another guard job
- [ ] exercise the decision logic against every branch with the two API calls stubbed
- [ ] CHANGELOG entry
