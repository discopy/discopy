# TODO.md

> Give this one a shot https://github.com/discopy/discopy/issues/472

- [WIP] @session_01RPJDHX3HZeqn2EP6wQQAtB-2026-07-26 21:16 Merge the symmetric-layer PR (#362) branch into this branch and resolve its conflicts with main
- [ ] Split `markov` into `markov` (comonoid: copy, discard) and `comarkov` (monoid: merge, unit)
- [ ] Add `finset.Function` and a `markov.Function` box holding the opposite of a function between finite sets
- [ ] Make `markov.Layer` alternate between function-opposites and generators, the same way `symmetric.Layer` alternates permutations and generators
- [ ] Add `frobenius.Cospan` and make `frobenius.Layer` alternate cospans and generators (see issue #472)
