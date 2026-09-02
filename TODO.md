# TODO

> can you push a branch (no need for a PR) on discopy with minimal snippets reproducing each bug, e.g. test/fable/{P1, P2, B39, ...}.py
>
> add links to the comment directly

> push the repro scripts as a test/fable branch like last time

- [x] One repro file per finding of the second read, `test/fable/B1.py` … `B45.py`, each asserting
  the correct behaviour so it fails while its bug is live and passes once fixed.
- [x] One miniature per property, `test/fable/P1.py` … `P9.py`, looping curated examples and
  reporting every violation — the sketch the property-based-testing suite generalises.
- [x] Verify every file is red for the right reason on `main` (`107c846`).
- [x] Link each bullet of the [#606 comment](https://github.com/discopy/discopy/issues/606#issuecomment-5371739173)
  to its file on this branch.
- [x] Merge `main` (`ce044a0`) and re-verify the first read's suite: 99 red of 104.
- [x] One repro file per finding of the third read, `test/fable/B46.py` … `B88.py`, and one
  miniature per new property, `P10.py` … `P13.py`, red for the right reason on `ce044a0`.
- [x] Link each bullet of [#699](https://github.com/discopy/discopy/issues/699) to its file on this branch.
