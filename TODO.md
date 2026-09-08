# TODO

Prompt, toumix on discopy#736, 2026-09-08 12:59 UTC, verbatim:

> forget about interaction and bidirectionality for now, focus on getting a feedforward neural network to compile from diagram to pytorch/hax

- [x] `Dims.check` and `Box.forward`: a layer's module applied to one tensor per leg, the shape of each tensor checked against its leg
- [x] `Network.to_function`: a feedforward network compiled to a `python.Function` on tensors by the functor sending legs to a tensor type and boxes to `forward`, with copy, discard and swap those of `python.Function`; a trace refuses
- [x] `discopy.neural.torch.Module`: the network as a PyTorch module, the modules of its boxes registered as submodules, `forward` running the compiled function, `torch.compile` taking it
- [x] JAX: `jax.jit` and `jax.grad` through the compiled function, tested
- [x] the package docstring, the changelog and the coverage omit for `torch.compile`'s writes
