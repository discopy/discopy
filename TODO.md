# TODO — Review restart for multiplicative hypergraphs

Alexis's live directive, verbatim:

> Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

## Checklist

- [WIP] @session_011Jk8KSTZk1eQoGgJAiEvEg-2026-08-14 17:10 Address the review on
  [PR #363](https://github.com/discopy/discopy/pull/363): specialize
  `hypergraph.Hypergraph[Function]`, translate source hypergraphs with a
  `hypergraph.Functor`, retain carry-save-adder coverage with public helpers,
  and verify messages, representation, equality, and lint.

## Mathematical description

A Python-valued hypergraph is the existing generic hypergraph data structure
whose boxes are multiplicative functions. Evaluation translates each source
box through a hypergraph functor, then executes the resulting causal network in
topological order; no second notion of hypergraph or spider wiring is needed.

## History

Restarted from current `main`; the published PR tip
`e3c073606a8f2714bb0afbe92f0c2875d80d9247` is retained as an ancestor by a
normal merge.
