# TODO

Prompt: [#482 "Bugs and hot spots in `discopy.quantum`"](https://github.com/discopy/discopy/issues/482)
by @giodefelice, nine items each with a reproducing snippet, plus @toumix's ruling on the
only contested one (verbatim, [2026-07-27](https://github.com/discopy/discopy/issues/482#issuecomment-5090765821)):

> The rest looks good but `Sqrt(-1)` should raise an error, as it's undefined which square root we mean.

Superseded on review of this PR ([2026-08-04](https://github.com/discopy/discopy/pull/517#discussion_r3713214000)), verbatim:

> ok I changed my mind let's do square roots of negative numbers, let's just fix the Sqrt.dagger method

So all nine items land as reported, item 5 included.

## Correctness

- [x] 1. `CQ.__str__` prints classical types as quantum — `channel.py:99` returns `Q(self.classical)` where it means `C(...)`
- [x] 2. `Measure(override_bits=True)` cannot be evaluated — `channel.py:320` passes a `Dim` to `Channel.discard`, which expects a `CQ`
- [x] 3. `Encode` ignores `constructive`/`reset_bits` in its types — `gates.py:193` hard-codes `bit ** n, qubit ** n`, so daggers are not type-correct
- [x] 4. `CRz`/`CRx`/`CU1` with `distance != 1` raise — `gates.py:495` calls `type(self)(controlled)` on a sugar class that takes a phase
- [x] 5. `Sqrt.dagger()` returned `self`, so `(Sqrt(-1) >> Sqrt(-1).dagger()).eval().array` was `-1` — it now conjugates, per @toumix's second ruling above
- [x] 6. `Channel.cups`/`Channel.discard` hard-code `Channel` instead of `cls`, so `Channel[float]` silently yields a `Channel[complex]`

## Performance

- [x] 7. `Channel.tensor` builds four boxes, two swap diagrams and a fresh `tensor.Functor` on every `@` — replace by the rank-0 `tensordot` plus one fixed interleaving permutation
- [x] 8. `Channel.measure` materialises `n**3`/`n**5` entries in a list comprehension for an array with `n` non-zeros — build the zeros and assign the diagonal
- [x] 9. `Circuit.measure(mixed=False)` runs `2**n` contractions where `np.absolute(self.eval().array) ** 2` gives the same result in one

## Wrap-up

- [x] Regression test for every item, `CHANGELOG.md` entry, `pflake8 discopy` and `coverage run -m pytest` clean

## Measured

`pflake8 discopy` clean, `coverage run -m pytest --skip-extra` 630 passed, 51 skipped.
The suite was also run once with `sympy` and `pytket` installed from PyPI (636 passed), so
`test/quantum/tk.py` and the sympy-gated `test/quantum/circuit.py` tests exercised these changes.

Item 7 was checked against the old implementation on 25 combinations of classical/quantum
domain and codomain shapes: identical arrays, `dom` and `cod` every time. Timings, best of
three on this machine:

| | old | new |
|---|---|---|
| one `Channel.id(Q(2)) @ Channel.id(Q(2))` | 13.5 ms | 0.11 ms |
| four-qubit nine-box `eval(mixed=True)` | 0.247 s | 0.021 s |
| `Circuit.measure()`, 8 qubits | 0.105 s | 0.018 s |
| `Circuit.measure()`, 10 qubits | 0.443 s | 0.024 s |

Item 8 is the one that did not land as reported. `Channel.measure` now builds its copy spider
via `Tensor.copy`, which is 2.6x at `n=24` destructive and 5.2x at `n=12` non-destructive — but
at `n=24` non-destructive it is a wash (0.84 s -> 1.01 s), because the array is 127 MB and
`Matrix.__init__` copies it a second time with `np.array(array, dtype=self.dtype)`. That copy,
not the list comprehension, is the remaining bottleneck; filed separately rather than changed
here, since making it `np.asarray` would alias caller arrays across the whole library.
