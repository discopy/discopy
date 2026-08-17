# TODO

> add a new function discopy.quantum.ansatze.Rydberg to simulate the QPU from
> pasqal https://docs.pasqal.com/qpu-emulators/emumps/advanced/hamiltonian/

- [x] `Rydberg` in `discopy.quantum.ansatze`: Trotterized time evolution under
  the Rydberg Hamiltonian, with atom positions, `duration`, piecewise-constant
  `omega`, `delta` and `phase` waveforms and the `C6` interaction coefficient
- [x] tests: exact diagonal evolution at `omega=0`, exact single-atom drive,
  Trotter convergence to the dense matrix exponential, waveform length checks
- [x] docs: Hamiltonian in the docstring with a link to Pasqal's page,
  autosummary entry, `CHANGELOG.md`
- [x] `pflake8` and the quantum test suite green

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

> merge the quantum reservoir PR into this one, add a Rydberg reservoir as example

- [x] merge #519's branch `claude/quantum-reservoir-discopy-pqbqgo`
  into this one, keeping both `TODO.md` sections
- [x] a Rydberg reservoir as example: a doctest where the `unitary` of a
  `Reservoir` is built by the `Rydberg` ansatz, with the
  :cite:`BravoEtAl22` entry in `docs/discopy.bib`
- [x] fix #587, found on the way: `Controlled._decompose` broke `eval` for
  `CU1`, `CRz` and `CRx` at any distance other than one — one line, with a
  regression test in `test/quantum/circuit.py`
