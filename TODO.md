# TODO

Prompt, verbatim:

> i feel like the setup with counterexamples is kind of botched. what is
> a good setup using hypothesis' GithubArtifactDatabase? how can i have
> a fast CI while also finding new counterexamples automatically?
> rethink the property testing infrastructure in a simple way, making
> maximal use of hypothesis' features.

> write his as a plan

> is there a way to generate tests without the globals()["test_{...}"]
> hack?

Decided with the maintainer: lands on this branch as one more round,
with a nightly explore run; no test generation, one conftest hook keys
the existing parametrized cells.

- [x] `proptest/conftest.py`: `pr`/`explore`/`dev` profiles with
      explicit `derandomize`/`database`, the local directory database,
      the read-only `GitHubArtifactDatabase` behind `GITHUB_TOKEN`, and
      the `pytest_runtest_setup` hook keying every Hypothesis cell by
      its node id.
- [x] `proptest/test_axioms.py` and the other property files: strict
      xfail for `.failing` cells, `note` of the verdict, per-test
      settings moved into the profiles.
- [x] `proptest/test_counterexamples.py`: strict xfail.
- [x] `discopy/testing.py`: `assert_axioms` replays the database
      (`Phase.reuse`) before generating — reconsidered and dropped:
      `find` keys the database by its predicate and saves whatever
      satisfies it, so a reuse phase would replay the same example
      forever rather than known counterexamples.
- [x] `.github/workflows/proptest.yml`: nightly schedule, `actions:
      read`, artifact download before the run and upload after it,
      profile chosen by event, rewritten concurrency comment.
- [x] PROPTEST.md, CONTRIBUTING.md, CHANGELOG.md, PR body; one PR
      comment explaining the redesign to the reviewers of the previous
      shape.
- [ ] Delete this file.
