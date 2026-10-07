> move the reusable parts of the trilingual parser away from lambek into a new discopy PR, i.e. without any experimental results or dataset adaptors, only the constructive linker, sinkhorn parser, Hungarian decoder etc

> take your time to make something beautiful

From [rel-int/lambek#8](https://github.com/rel-int/lambek/pull/8), on top of `grammar.abstract` (#400).

- [WIP] @session_01SRPjfa1mCGgEnfpQuTsP3F-2026-10-07 08:35 `grammar.proofnet`: the ports of a type, `ProofNet.from_term` (the axiom linking of a linear term), `ProofNet.to_term` (sequentialisation, which is the correctness criterion), `ProofNet.decode` with the Hungarian algorithm
- [ ] `grammar.neural`: `Signature` (types in Polish notation), the constructive `Tagger`, `sinkhorn`, the `Linker`, a `Parser` with one loss and one `parse`
- [ ] tests, API docs, CHANGELOG, `--skip-extra` for a module that imports torch
- [ ] `pflake8`, `pylint` at or above `fail-under`, coverage
