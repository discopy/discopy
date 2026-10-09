# TODO

Human prompt, verbatim:

> ok do it, then resolve commits and push

Review round on [#730](https://github.com/discopy/discopy/pull/730):

> **@0x0f0f0f**, on `discopy/cat.py:395`: agent removed. restore

- [x] Restore the "setoid hell" warning in the docstring of `Arrow.setoid`
- [x] Drop the unused loop variables of `EHypergraph.costs` and
      `EHypergraph.section`, which fail the `lint` job
- [x] Merge `main` into the branch, resolving `CHANGELOG.md`
- [x] `pflake8`, `pylint`, `pytest`
- [x] Reply to and resolve the outdated review threads of the last round
