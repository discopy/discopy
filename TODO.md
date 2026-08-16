# TODO

> implement a quantum reservoir model with discopy

- [x] `discopy.quantum.reservoir` with a `Reservoir` class:
  a fixed unitary on memory and input qubits, one time step as a channel
  (encode the input, apply the unitary, discard the input qubits), features
  as the Born probabilities of the memory qubits, ridge-regression readout
- [x] tests in `test/quantum/reservoir.py`
- [x] docs: list the module in `docs/api/quantum.rst`, add a `CHANGELOG.md` entry
- [x] `uv run pflake8 discopy` and `uv run coverage run -m pytest --skip-extra` green
  (626 passed, 51 skipped)
- [x] address the cubic review: validate `fit` inputs, exact test command in `TODO.md`
