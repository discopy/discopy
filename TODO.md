# TODO

Review round, [cubic on #641](https://github.com/discopy/discopy/pull/641), P1:

> When multiple commits arrive while a run is active, this group can discard the synchronize
> event that deleted `TODO.md`; the surviving run then compares against a no-TODO predecessor
> and leaves the PR draft. Make deletion detection recover skipped synchronize events instead
> of relying on every event being retained.

- [ ] drop the concurrency group, which cannot keep a pending run alive
- [ ] skip the draft branch on an event the head has already moved past, so ordering stops mattering
- [ ] extend the harness with the superseded cases
- [ ] CHANGELOG
