# TODO

> add a new function discopy.quantum.ansatze.Rydberg to simulate the QPU from
> pasqal https://docs.pasqal.com/qpu-emulators/emumps/advanced/hamiltonian/

- [WIP] @t9pcwe-2026-08-17 12:00 `Rydberg` in `discopy.quantum.ansatze`: Trotterized time evolution under
  the Rydberg Hamiltonian, with atom positions, `duration`, piecewise-constant
  `omega`, `delta` and `phase` waveforms and the `C6` interaction coefficient
- [WIP] @t9pcwe-2026-08-17 12:00 tests: exact diagonal evolution at `omega=0`, exact single-atom drive,
  Trotter convergence to the dense matrix exponential, waveform length checks
- [WIP] @t9pcwe-2026-08-17 12:00 docs: Hamiltonian in the docstring with a link to Pasqal's page,
  autosummary entry, `CHANGELOG.md`
- [WIP] @t9pcwe-2026-08-17 12:00 `pflake8` and the quantum test suite green
