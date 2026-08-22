# TODO

> I want to setup a reviewer that checks for style, keeping cubic only for correctness. Something like “read the whole file and make sure your local diff is consistent” plus the STYLE.md of DisCoPy. Probably it should be built into an app rather than using my agents handle, wdyt?

Design settled in the same session: reviews are posted by the existing
discopy-bot GitHub App, the harness is Claude Code running on the GitHub
runner as one replaceable workflow step (“DeepSeek harness is probably too
unstable atm but it shouldn’t be hard to switch later?”), and inference goes
through USER's OpenModel gateway with the model pinned by a repository
variable (“I’d rather use an open weights model”). “build the whole thing,
starting from the todo as usual”

- [x] `.github/workflows/style-review.yml`: runs when a pull request leaves
  draft or gets the `style-review` label, same-repo pull requests only,
  with concurrency, timeouts and a graceful skip while unconfigured
- [x] `.github/style-review/prompt.md`: read every changed file whole, check
  the diff against the file's own conventions and `STYLE.md`, leave
  correctness to cubic and linting to pflake8, findings as JSON
- [x] `.github/style-review/post.py`: validate the findings and post one
  discopy-bot review, inline where the diff allows and in the body
  otherwise, posting nothing when the diff is clean
- [x] `CHANGELOG.md` entry
- [ ] USER: create the `OPENMODEL_API_KEY` secret and the
  `STYLE_REVIEW_BASE_URL` / `STYLE_REVIEW_MODEL` repository variables from
  the OpenModel console
- [ ] run the reviewer on a real pull request and read its first review
