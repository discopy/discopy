in a new side-branch based on property testing (current branch), analyze the bugs found in https://github.com/discopy/discopy/issues/606 and come up with a list of property tests we could add (besides axioms) that would reproduce this bug. categorize every failure by the property being violated or otherwise when the failure is very specific and doesn't really violate a clearly stated expected property, defer it to the unit test suite.

# Work

- [WIP] @opencode606-2026-08-24 10:25 Analyze the concrete bugs referenced by issue #606.
- [ ] Classify each failure under a general property or defer it to the unit test suite.
- [ ] Record a concise list of proposed non-axiom property tests and verify the classification.
