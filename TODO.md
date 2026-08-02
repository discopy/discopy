# TODO

Prompt: [#482 "Bugs and hot spots in `discopy.quantum`"](https://github.com/discopy/discopy/issues/482)
by @giodefelice, nine items each with a reproducing snippet, plus @toumix's ruling on the
only contested one (verbatim, [2026-07-27](https://github.com/discopy/discopy/issues/482#issuecomment-5090765821)):

> The rest looks good but `Sqrt(-1)` should raise an error, as it's undefined which square root we mean.

So items 1–4 and 6–9 land as reported; item 5 is resolved by rejecting the input rather than
fixing the dagger.

## Correctness

- [ ] 1. `CQ.__str__` prints classical types as quantum — `channel.py:99` returns `Q(self.classical)` where it means `C(...)`
- [ ] 2. `Measure(override_bits=True)` cannot be evaluated — `channel.py:320` passes a `Dim` to `Channel.discard`, which expects a `CQ`
- [ ] 3. `Encode` ignores `constructive`/`reset_bits` in its types — `gates.py:193` hard-codes `bit ** n, qubit ** n`, so daggers are not type-correct
- [ ] 4. `CRz`/`CRx`/`CU1` with `distance != 1` raise — `gates.py:495` calls `type(self)(controlled)` on a sugar class that takes a phase
- [ ] 5. `Sqrt` of a negative or complex number raises, per @toumix's ruling above
- [ ] 6. `Channel.cups`/`Channel.discard` hard-code `Channel` instead of `cls`, so `Channel[float]` silently yields a `Channel[complex]`

## Performance

- [ ] 7. `Channel.tensor` builds four boxes, two swap diagrams and a fresh `tensor.Functor` on every `@` — replace by the rank-0 `tensordot` plus one fixed interleaving permutation
- [ ] 8. `Channel.measure` materialises `n**3`/`n**5` entries in a list comprehension for an array with `n` non-zeros — build the zeros and assign the diagonal
- [ ] 9. `Circuit.measure(mixed=False)` runs `2**n` contractions where `np.absolute(self.eval().array) ** 2` gives the same result in one

## Wrap-up

- [ ] Regression test for every item, `CHANGELOG.md` entry, `pflake8 discopy` and `coverage run -m pytest` clean
