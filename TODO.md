# TODO

Prompt ([#380](https://github.com/discopy/discopy/issues/380), verbatim):

> If we add an attribute `Diagram.functor_factory` then we don't need `Diagram.hypergraph_factory` nor `Hypergraph.functor`  anymore because we can have `Hypergraph(NamedGeneric["category"])` access that functor factory. In fact, we shouldn't need a new `Hypergraph` class in each module anymore, `monoidal.Hypergraph = Hypergraph[Diagram]` would be enough.
>
> The only change is to move the `make_causal_first` logic from `markov.Hypergraph` back to the main `Hypergraph.to_diagram` which would just check whether the domain of the functor `self.category` is a subclass of `MarkovCategory` (the other `monogamous` check in `monoidal.Hypergraph` is superfluous).
>
> The end goal is that the hypergraph equality should work for custom subclasses of `Diagram` without having to create a dedicated `Hypergraph` class, see e.g. the example from the readme in #378.

---

- [ ] Add `Diagram.functor_factory` and have `Hypergraph[Diagram]` access it, replacing `Diagram.hypergraph_factory` and `Hypergraph.functor`
- [ ] Replace the per-module `Hypergraph` classes by `Hypergraph[Diagram]` (e.g. `monoidal.Hypergraph = Hypergraph[Diagram]`)
- [ ] Move the `make_causal_first` logic from `markov.Hypergraph` into `Hypergraph.to_diagram`, gated on `self.category` domain being a `MarkovCategory`; drop the superfluous `monogamous` check in `monoidal.Hypergraph`
- [ ] Test: hypergraph equality works for a custom `Diagram` subclass without a dedicated `Hypergraph` class (README example from #378)
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`
