# TODO

> Let's improve the style reviewer so that it posts one message per iteration
> then when new comments or commits come in it decides whether the reviews were
> taken into account and edits its first message with "X+Y style remarks taken
> into accounts: X accepted / Y declined"

- [x] Mark every review the style reviewer posts, so a later round can read
      back the remarks it made
- [x] Read the previous rounds and the replies they got into
      `.style-review/history.json`
- [x] Ask the model, in the same one request that reviews the revision, for a
      verdict on each past remark: accepted, declined or still open
- [x] Edit the first review's body with the running tally
- [x] Document the change in `CHANGELOG.md`
