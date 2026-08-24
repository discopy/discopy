# TODO

> I want to turn this kind of deep read-through of the whole codebase into a workflow I can trigger from github directly https://github.com/discopy/discopy/issues/606

> let's also add a label to the issues so that before scanning the codebase and opening a new issue, the agent can have a look at the results of the previous runs

- [x] `.github/workflows/codebase-read.yml`: a `workflow_dispatch` workflow running the read with `anthropics/claude-code-action`, skipping with a notice until the `ANTHROPIC_API_KEY` secret is set next to the discopy-bot app of #608
- [x] `.github/codebase-read/prompt.md`: the read instructions — previous `codebase-read` issues first as baseline, every module whole in one sitting, strains ranked by leverage, a bug reported only with an executed repro
- [x] `.github/codebase-read/post.py`: open the report as a new `codebase-read`-labelled issue and the verified bugs as its first comment, as discopy-bot
- [x] `CHANGELOG.md` entry and `.codebase-read/` in `.gitignore`
