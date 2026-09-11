# TODO

> #658 finally got merged. do a global update with all recent changes in CHANGELOG.md

#658 was **squash-merged** to `main` as `86c15ee`, `split/1-axiom-infra` was
deleted and #659 auto-retargeted onto `main`, which left it `dirty`: its
history carries #658's unsquashed commits while `main` carries the squash.

- [ ] Converge this branch onto #659's head `76b8f32`, whose resolutions
      supersede this branch's own
- [ ] Merge `main` (`86c15ee`, the #658 squash) and resolve the conflicts
- [ ] Rewrite `CHANGELOG.md`'s `[Unreleased]` so it reads as one section
      covering everything released since `1.2.2`, not two merged drafts
- [ ] Correct the feedback-unroll citation `#606` -> `#649` wherever it appears
- [ ] `pflake8` clean, both suites green, counts explained
