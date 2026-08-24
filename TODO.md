in a new side-branch based on property testing (current branch), analyze the bugs found in https://github.com/discopy/discopy/issues/606 and come up with a list of property tests we could add (besides axioms) that would reproduce this bug. categorize every failure by the property being violated or otherwise when the failure is very specific and doesn't really violate a clearly stated expected property, defer it to the unit test suite.

have a look at https://github.com/discopy/discopy/tree/claude/discopy-codebase-review-nx03n1/test/fable, it contains self-contained bug reproductions. rebase on this branch and augment them with a property test reproducing the bug. don't reproduct these tests yourself

implement it

# Current Work

- [x] Analyze the concrete bugs referenced by issue #606.
- [x] Classify each failure under a general property or defer it to the unit test suite.
- [x] Augment eligible fable repros with Hypothesis properties without duplicating them.
- [x] Record the resulting non-axiom property list and verify the suite.

# Semantic Property Tests

- [ ] Add constructive semantic carrier and morphism strategies.
- [ ] Bind declared semantic capabilities to the axiom matrix.
- [ ] Add boundary, equality, closure and bug-coverage tests.
- [ ] Run property-test lint and verification.

# Imported Harness

> can you push a branch (no need for a PR) on discopy with minimal snippets reproducing each bug, e.g. test/fable/{P1, P2, B39, ...}.py
>
> add links to the comment directly

- [x] One repro file per finding of the second read, `test/fable/B1.py` … `B45.py`, each asserting
  the correct behaviour so it fails while its bug is live and passes once fixed.
- [x] One miniature per property, `test/fable/P1.py` … `P9.py`, looping curated examples and
  reporting every violation — the sketch the property-based-testing suite generalises.
- [x] Verify every file is red for the right reason on `main` (`107c846`).
- [x] Link each bullet of the [#606 comment](https://github.com/discopy/discopy/issues/606#issuecomment-5371739173)
  to its file on this branch.
