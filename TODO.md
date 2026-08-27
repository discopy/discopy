# TODO

> read our discopy workflows and propose some refactors and improvements, start by reading the related issues and PRs

> fold all your refactors into a PR
> 7. why javascript? can it be Python?
> merge base sounds more coherent than main tip

- [ ] 1. Lint and test the workflows and the code inside them
- [ ] 2. Stop cancelling `main`'s runs in `build.yml` and `benchmark.yml`
- [ ] 3. Give `build.yml` a `permissions:` block
- [ ] 4. Drop the dead `SRC_DIR`/`TEST_DIR` and the gone `tooling/uv-migration`
- [ ] 5. One composite action for the repeated setup preamble
- [ ] 6. Pin every action by SHA, add `dependabot.yml`
- [ ] 7. Port `benchmark-comment.yml`'s JavaScript to Python
- [ ] 8. Benchmark against the merge base, not the base tip
- [ ] 9. `CHANGELOG.md` entry
- [ ] 10. Green CI on the merged tree
