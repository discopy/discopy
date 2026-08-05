# TODO

Prompt (🌙 Evening, scheduled 2026-08-05, verbatim):

> Follow toumix/desire/EVENING.md

Picked [#528](https://github.com/discopy/discopy/issues/528) because it is the only
open issue small enough for one night that lands in a file **no** queued PR holds
open: the eleven ready PRs touch `symmetric.py`, `closed.py`, `quantum/`,
`drawing/` and their baselines, none of them `grammar/categorial.py`. Issue body:

> Two corners of `grammar.categorial.cat2ty`, both hit by real CCGbank/rebank categories:
>
> ```python
> >>> cat2ty(r"(S\NP)")
> Ty(Ob('(S\\NP)'))          # atom named "(S\NP)", expected Ty('NP') >> Ty('S')
> >>> cat2ty(r"(S\NP)\(S\NP)[conj]")
> ... >> Ty(Ob('NP)[conj'))  # atom named "NP)[conj"
> ```
>
> Cause: `split` only unbrackets the operands around a top-level slash, so a fully
> parenthesized category falls through to the atom branch, and `remove_modifier` only
> fires in that branch, so a feature on a complex category (CCGbank writes coordination
> as `X[conj]` for any `X`) is never stripped and splits mid-atom.
>
> Workaround used in rel-int/rebank `experiments/bridge.py`: strip `\[[^]]*\]` globally
> and unwrap outer brackets before calling `cat2ty`.

---

- [x] @evening-2026-08-05T00:15 A fully parenthesized category unwraps instead of
      becoming an atom
- [x] @evening-2026-08-05T00:15 A feature is stripped wherever it occurs, not only on
      an atom
- [x] @evening-2026-08-05T00:15 Slashes associate to the left, as in Steedman's
      convention — a third bug in the same three lines, found while fixing the two
      above and not part of the issue, see the note below
- [x] @evening-2026-08-05T00:15 Tests for all three, plus a doctest so the convention
      is documented where it is implemented
- [x] @evening-2026-08-05T00:15 `CHANGELOG.md` entry under `[Unreleased]`
- [x] @evening-2026-08-05T00:15 `pflake8 discopy` and `coverage run -m pytest`

## Left associativity (🌙 evening, 2026-08-05)

`split` returned at the **first** top-level slash, so `S\NP/NP` came out as
`S\(NP/NP)` — `((NP << NP) >> S)` — where Steedman's convention, which CCGbank and
depccg both follow, reads it as `(S\NP)/NP`. It is not in #528 because CCGbank writes
complex operands fully bracketed, so the corner is only reachable from hand-written
input; the whole fix is scanning the string in reverse, and leaving it would have left
the reader of the new `split` to re-derive why it scans forwards.

Nothing in the repo pinned the old reading: `cat2ty` had no test and no doctest before
this branch, and every category in `test/grammar/categorial.py`'s depccg fixture is
fully bracketed, so it parses identically either way.

## Verification (🌙 evening, 2026-08-05)

`uv run pflake8 discopy` clean. `uv run coverage run -m pytest`: **592 passed, 83
failed, 6 skipped** — the same 83 on this branch and on `main` `e80ea38`, compared as
sorted `FAILED` lists and diffed to nothing, so none of them is this change. They are
the container's missing optional stack (`nltk`, `tensornetwork`, `jax`, …) plus the
version skew that ad-hoc extras give, exactly as recorded on
[#499](https://github.com/discopy/discopy/issues/499); the extras that do install were
installed targeted (`pennylane`, `pytket`, `pyzx`, `sympy`, `torch`) to get collection
past `discopy/quantum/{tk,pennylane}.py`. Zero failures anywhere under `categorial`.

## Review (🌙 evening, 2026-08-05)

daydream6728 approved and suggested making the docstring raw so the CCG categories in
it need no escaping. Applied in the commit above: `r"""`, and every `\\` in the
examples is now the single `\` a user would actually type. No behaviour change —
`pflake8` clean, the doctest and `test/grammar/categorial.py` still pass, 11 passed.

The thread is left open and unanswered: `RULES.md` rule 4 only lets an agent reply to
another user's comment once USER has replied or marked it with a 🚀.
