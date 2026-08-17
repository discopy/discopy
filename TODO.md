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
