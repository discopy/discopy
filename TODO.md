# TODO

USER's review on #701 (docstring restructuring + two questions), 2026-09-07:

> what you give here are not axioms but definitions, they should move to the docs for the
> different methods (then,tensor,etc) not in the module docstring

(module docstring, "Axioms" section)

> Again this is not really an example but a definition of another construction which happens
> to be a special case of the first one, split the docs into the different methods of Lens

(module docstring, "Example" section)

> You're getting ahead of yourself, this should be just a link to a module discopy.neural
> where this sentence is explained in more detail

(the "A neural network is..." paragraph)

> I'd rather avoid that special case by waiting for the PRO -> Nat rename PR to be merged

(`Ty.__str__`'s `except TypeError`)

> high-level question: does the category of optics inherit any structure from its base
> further than symmetric? it feels a bit suspect that we needed many para constructions but
> optics and lens only have one?

(design question, answer as a reply, no code change expected)

> looks amazing but why all this extra margin on top and bottom?

(tensor.svg question, answer as a reply after measuring)

- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 19:15 Move the composition/tensor/id/swap doctests (and the `then.svg`/`tensor.svg` images) out of the module "Axioms" section into `Optic.then`, `Optic.tensor`, `Optic.id`/`Optic.swap`.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 19:15 Trim the module "Example" to the one true usage example (the pair accessor); move the reverse-derivative definition and its doctest into `Lens.then`.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 19:15 Replace the neural-network paragraph with one link to `discopy.neural`.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 19:15 Drop `Ty.__str__`'s `except TypeError` special case.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 19:15 Answer the design question and the margin question as replies; no code change expected from either.
