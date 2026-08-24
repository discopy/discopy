# TODO

cubic's review of 2026-08-24 on the compact-terms PR (#489):

> P2: When a replacement contains a variable bound by `Let`, this branch performs variable capture and changes the term's semantics. Alpha-rename conflicting bound variables before substituting into the body.

- [x] Guard `Substitution` against variable capture under a binder, in the `Let` and the pre-existing `Abstraction` branch alike: raise a `ValueError` telling the user to rename, like `Let.__init__` already does on clashing variables, rather than adding alpha-renaming machinery to a PR already at the size red line.
