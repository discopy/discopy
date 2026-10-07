> add a follow up PR with notebook that showcases the new module with parsing of arithmetic formulae as example, i.e. recover the bracketed a + (b * c) from the sequence a + b * c

Stacked on #798 (`grammar.proofnet` and `grammar.neural`).

- [WIP] @session_01SRPjfa1mCGgEnfpQuTsP3F-2026-10-07 09:40 `docs/notebooks/parsing-arithmetic.md`: formulae as terms, the two readings of `a + b * c` as two linkings of one sequent, gold nets by precedence climbing, a contextual encoder with `Parser`, training, held-out accuracy and longer formulae
- [ ] the notebook in the docs toctree, CHANGELOG
- [ ] `export_notebooks.py --check` runs it, under a minute on CPU
