# TODO

Human prompt, verbatim:

> ok now we need to target the comments in https://github.com/discopy/discopy/pull/730/  - first of all we need to rename Carrier and Table to EHypergraph, then we need to go through the issues one by one, you need to explain them to me like and /skill:caveman  and we need to evaluate all possible solutions and alternatives that we have and their tradeoffs and then we decide what to do

Review round on [#730](https://github.com/discopy/discopy/pull/730):

> **@toumix**, on `discopy/table.py:6`: if a carrier is a table then let's
> call it a table? "carrier" is a meaningless term i'd rather avoid it
>
> **@daydream6728**: what about `EHypergraph`?

> **@toumix**, on `discopy/table.py:593`: [...] IMO, Morphism and Carrier
> should be fused into one class which could be called `EGraph`. [...] it's
> fine if `EGraph` isn't a mere egraph but an "open egraph" i.e. with
> designated dom and cod.
>
> **@toumix**, on `discopy/table.py:539`: [...] The objects of your category
> of egraphs shouldn't need to contain the data for the morphisms of which
> they are the domain and codomain.
>
> **@toumix**, on `discopy/table.py:221`:
> `class EGraph(MonoidalCategory, NamedGeneric['category']):`

> **@daydream6728**, on `discopy/table.py:462`: Why this restriction? would
> the tabular storage not work for traced categories? [...] do we expect
> caps, cups, spiders and swap to be generators here instead of being
> encoded as wiring structure? [...] if we implemented conversion from a
> `hypergraph.Hypergraph[Diagram]` instead, we'd actually have a choice in
> what we encode by wiring [...]

> **@daydream6728**, on `discopy/table.py:656`: could define a setoid like
> cat.Arrow, which would automatically derive __eq__ and __hash__? [...] In
> any case i think its fine to define __eq__ as equiv, where would we need
> strict equality?

> **@daydream6728**, on `discopy/table.py:522`: could inherit from Monoid to
> automatically have __matmul__ = tensor, and from NamedGeneric to enforce
> more faithfully the homogeneity constraint when tensoring

> **@daydream6728**, on `discopy/table.py:153`: could be a monoid too, where
> tensor calls to append

> **@daydream6728**, on `discopy/utils.py:427`: could we use union by rank
> here?

- [x] Rename `discopy.table` to `discopy.ehypergraph` and `Carrier` to
      `EHypergraph`
- [ ] Decide and apply: fuse `Morphism` into `EHypergraph` (open e-hypergraph)
- [ ] Decide and apply: structural morphisms as wiring vs generators
- [ ] Decide and apply: `__eq__` as `equiv`, or a setoid
- [ ] Decide and apply: `Wires` as a `Monoid`
- [ ] Decide and apply: `Shard` as a `Monoid`
- [ ] Decide and apply: union by rank in `UnionFind`
- [ ] Merge `main` into the branch, `pflake8`, `pytest`, coverage, sphinx
