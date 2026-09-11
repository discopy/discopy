# TODO

> * remove the entire counterexamples mechanism, let's just trust hypothesis to generate a counterexample on demand
> * remove the big module-level docstring that explains how to use proptesting, i trust agents to be smart enough to guess how to make good use of it from the instructions in CONTRIBUTE.md and class-/method-level docstrings. instead of explaining a whole workflow, just provide a normal module docstring explaining what this module does like every other module

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-11 09:05 Delete `proptest/test_counterexamples.py` and every trace of the recording protocol; `Axiom.falsify` is how a counterexample is produced on demand
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-11 09:05 Cut the module docstring of `discopy.axioms` down to what the module is, keeping the autosummary as every other module has it
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-11 09:05 Follow through in `CONTRIBUTING.md`, `AGENTS.md` and the changelog; run the checks, push, tell #659, refresh the PR body
