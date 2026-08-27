# TODO

> ha the base commit moving is annoying, we don't wanna throw the review and
> start again but we don't want to give a stale review either, let's do
> something in between: we give the review back to the style reviewer with the
> diff wrt the new base and ask to update the line numbers

The base advancing turns out to move nothing: the merge base of a branch and
its target does not change when the target gains commits, so the three-dot
diff GitHub shows — and the one we compute — is the same before and after.
What does go stale is the head moving under a running round, and that push
already has a round of its own.

- [ ] The concurrency group drops `github.event.action`, so a newer trigger
      cancels an in-flight round whatever started either of them
- [ ] A round that finds the head has moved posts nothing, leaving the review
      to the round that push starts
