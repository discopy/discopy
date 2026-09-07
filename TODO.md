# TODO

> now we need to make a discopy PR as draft that contains the @../python/metatheory-refresh/ INITIAL data structure table in a module and a basic functor that can go back and forth from frobenius. /plan read in full the metatheory refresh repo, its docs and we design what to port properly, use /ponytail:ponytail when needed and we need to design a solid, tabular e-hypergraph data structure. one of the points should be that diagrams should and can be represented as tables. what would be the simplest, easiest solution ? the one that we have or others? evaluate all other different solutions and their pros and cons, then /mattpocock-skills:grilling and then we plan and execute the creation of a @discopy/ pull request off latest main.
>
> we don't want ECMap, but /mattpocock-skills:grilling properly on how to represent the table and what each feature of metatheory-refresh does and how do we get a basic initial version with roundtrip

Design decisions taken in the session, for reviewers:

* The store is the metatheory carrier: shards keyed `(op, n_in, n_out)` with
  numpy `int64` columns, wires in a union-find, one hashcons per shard.
* Arrows are boundary handles into a mutable store, as `Carrier`/`Morphism` are
  in metatheory, not whole immutable values as `Hypergraph`/`CMap` are.
* Spiders stay cells and lowering is a hand-written frame scan, both faithful to
  `metatheory/functors/diagram_carrier.py`.
* At the `MONOIDAL` point of metatheory's enrichment lattice: no absorption, no
  `(sign, value)` labels on the union-find, no canonizers.
* Each tree of the union-find is a vertex — the "meta-spider" whose legs are its
  member wires. No `⊥` row and no node-kind column: metatheory's own
  `docs/SEMANTICS.md` (branch `claude/semantics-md-review-ovxmpx`, commit
  `1d90a812`) demotes Tiurin's ⊥ e-box to RESERVED, because the flat
  nondeterministic join is already inductive in the union-find.

- [x] `discopy/table.py`: `SymbolTable`, `UnionFind`, `Shard`
- [x] `discopy/table.py`: `Carrier` (intern, hashcons, merge, rebuild, scan)
- [x] `discopy/table.py`: `Wires` and `Morphism`, the category structure
- [x] `discopy/table.py`: `Carrier.from_diagram`, the frame scan
- [x] `discopy/table.py`: `Morphism.to_diagram`, min-cost section then layout
- [x] `test/table.py` and the doctests
- [x] `discopy/__init__.py`, `docs/api/syntax.rst`, `CHANGELOG.md`
- [x] `pflake8`, `pylint`, `pytest`, coverage >= 98, sphinx
- [x] Fill in the pull request number in the `CHANGELOG.md` entry

Checks run locally: `pflake8 discopy` clean, `pylint discopy` 8.63/10,
`pytest` 954 passed 1 skipped with every extra installed, `coverage` 98% in
total and 100% on `discopy/table.py`, `sphinx-build` clean for the new pages.
