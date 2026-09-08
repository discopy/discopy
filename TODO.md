# TODO

Review round on #736, toumix on 2026-09-08 09:03–09:09 UTC, verbatim:

> this module should be called discopy.neural.network

> This example is useless remove it

> Let's redesign this altogether: we want a module for standard feedforward neural networks distinct from the free compact category it generates.

> this feels like a very annoying limitation: it should be possible to define a neural net that has different inputs and outputs

> let's make graph neural nets come in a later PR

> Network should be its own traced category of feedforward neural nets, i.e. it should have methods of composition, tensor, etc.
>
> Then CMap is the free compact category it generates

> let's make the PR readable on its own and avoid adding docs for modules that are further down the stack

- [ ] `discopy.neural.core` is renamed `discopy.neural.network`
- [ ] the example of the package docstring is removed
- [ ] the package docstring names what this pull request has and nothing further down the stack
- [ ] `Network` is a feedforward net `Dim -> Dim` with inputs and outputs of any widths, and networks form the free traced cartesian category on such boxes: `then`, `tensor`, `id`, `swap`, `copy`, `discard` and `trace` as diagram operations (the design is proposed on the pull request and waits on a 🚀)
- [ ] the compact category is `Int(Network)`: objects are pairs `(X+, X-)` of dimensions, a box `(X+, X-) -> (Y+, Y-)` is a network `X+ @ Y- -> Y+ @ X-`, `CMap` is its combinatorial maps with a routing per channel, and a feedforward net is the box `(X, 0) -> (Y, 0)`
- [ ] graph neural networks — the bidirectional cells `(X, X) -> (Y, Y)`, `mem`, signatures, `interpret`, `MapNN` — leave this stack for a later pull request
- [ ] #737, #738 and #705 are re-cut on the new base once it lands; #739, #740 and #741 wait for the later pull request
