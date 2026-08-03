# TODO

> implement a quantum reservoir model with discopy

- [WIP] @pqbqgo-2026-08-03 12:52 `discopy.quantum.reservoir` with a `Reservoir` class:
  a fixed unitary on memory and input qubits, one time step as a channel
  (encode the input, apply the unitary, discard the input qubits), features
  as the Born probabilities of the memory qubits, ridge-regression readout
- [ ] tests in `test/quantum/reservoir.py`
- [ ] docs: list the module in `docs/api/quantum.rst`, add a `CHANGELOG.md` entry
- [ ] `uv run pflake8 discopy` and `uv run coverage run -m pytest` green
