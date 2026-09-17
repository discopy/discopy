# TODO

> fix those two as well

The two docs breakages reported at the end of the previous round, both
`CRITICAL`s that silently drop content from the built pages:

- [x] the architecture table is missing from `discopy.abc`: `.. raw:: html
      :file: api/architecture.html` in the module docstring is resolved
      against `discopy/`, and the table's own links are written relative to
      the docs root, so it only works on a root-level page
- [ ] `discopy.hopf`'s whole `Axioms` section is missing: napoleon turns the
      `Example` section into an admonition and indents everything after it,
      including the `Axioms` title, into the box
