# Term-to-diagram roundtrip

> Term-to-diagram roundtrip https://github.com/discopy/discopy/pull/375
>
> the original PR was crap start again from scratch

Closes #372. Supersedes #375, which is abandoned: this branch starts from `main` and shares
no commits with it.

## The naming decision is ruled: B

Asked on this PR as A (canonical naming, no `varname`) or B (`varname` excluded from equality).
USER, [2026-08-13](https://github.com/discopy/discopy/pull/540#issuecomment-5277710609), verbatim:

> B

So: **`varname` goes on the wire, and is excluded from `__eq__`/`__hash__`.** The roundtrip is
faithful on the nose in both directions. The cost, accepted with the ruling, is a field equality
ignores — two `==` diagrams can produce two different terms. That has to be documented and pinned by
a test rather than left to be discovered.

The global counter of #372 is out regardless: `STYLE.md`'s determinism clause rules it out, and
`to_term` called twice would give different names. Fresh names are derived from binder position
instead (point 4), which is what plan A would have used everywhere and is now the fallback for
unannotated wires.

## What is actually true today

Measured, not assumed:

- There is no `to_term` anywhere in `discopy/`. The only occurrence of `varname` is a local variable
  in `biclosed.Ty.__call__`.
- `eval` quotients by alpha: alpha-equivalent terms give `==` diagrams. This is what B preserves and
  the discarded third reading would have broken.

### Where `varname` has to live, measured

This is the part that decides the size of the change, so it was measured rather than guessed:

- `biclosed.Ty('X').inside` holds a **`monoidal.Wire`**, not a `biclosed.Ob`. Only exponents are
  `biclosed.Ob` (`Over`/`Under`). So the annotation site is `Wire`, the atom of every `Ty` in the
  library — not a biclosed-local class.
- **Subclassing is not viable.** `Wire.__eq__` is `type(self) is type(other) and (name, dom, cod) ==
  ...` and `__hash__` includes `type(self)`, so an annotated subclass compares unequal to a plain
  atom *in both directions*, with a different hash. A subclass would silently split every type.
- A plain field on `Wire` itself does work, because `__eq__`/`__hash__` never mention it:

  ```
  a, b = Wire('X'), Wire('X'); b.varname = 'x'
  a == b            # True
  hash(a) == hash(b)  # True
  Ty(a) == Ty(b)    # True
  ```

So B is a root-level change to `monoidal.Wire`, kept invisible to equality. Two places already drop
the field and need to carry it: `Wire.dagger()` rebuilds via `type(self)(name, cod, dom, ...)`, and
`Wire.__repr__` returns `cat.Ob('X')` for a white/white wire.

### The statements to be tested

`to_term` is partial, total on the image of `eval`:

- **T1** — `t.eval().to_term().eval() == t.eval()`, for every term. Total.
- **T2** — `t.eval().to_term() == t`, for every term, **on the nose**. This is what B buys and what
  A could not give.
- **T3** — `to_term` raises a clear error off the image of `eval`.
- **T4** — annotations are invisible to equality: two diagrams differing only in `varname` are `==`
  with equal hashes, yet `to_term` gives different terms. Pins the accepted cost.
- **T5** — a diagram with no annotations still round-trips, with position-derived names.

### Terms are already a category, so `to_term` should be a functor

`TermBase` subclasses `Box` and already carries `dom` (the tensor of its free-variable types) and
`cod` (its type) — the shape of a morphism `context -> type`. Making that explicit gives `to_term`
through the existing `Functor`:

| categorical structure | term |
|---|---|
| `id(A)` | `Variable`, named from `varname` if present |
| `f >> g` | substitution |
| `f @ g` | context concatenation, renaming apart |
| `ev` | `Application` |
| `curry` | `Abstraction` |

`Functor` already walks layers, so the layer decomposition that made #375 sprawl is never written by
hand. Same move as `discopy.drawing`, where the layout algorithm is itself a functor.

## Points

- [x] 1. Get the naming decision ruled. B, quoted above.
- [x] 2. File the bugs found while measuring: #541, #542, #543, #544 — all four fixed in #545.
      #548 (no closed diagram containing `Copy` can be drawn) and #549 (`Context.dom` repeated
      #542's unbound call) also filed; #549 is fixed on `main` via #556.
- [x] 3. `varname` on `monoidal.Wire`, per USER's ruling quoted below: an optional argument of
      `Wire.__init__`, in `__repr__` but not `__str__`, not in `__eq__`. Also carried through
      `dagger()` and `to_tree`/`from_tree`, which drop it today.
- [ ] 4. Deterministic fresh names from binder position (de Bruijn level: free variables numbered by
      their index in `dom`, binders continuing the numbering), used wherever `varname` is absent.
      Test that it is a pure function of the diagram.
- [ ] 5. Annotate at the two sites #372 names: the identity wire built by `Variable.eval`, and the
      abstracted wire of the `Curry` built by `Abstraction.eval`.
- [ ] 6. Make terms-in-context a `BiclosedCategory` instance: `id`, `then`, `tensor`, `ev`, `curry`.
- [ ] 7. `biclosed.Diagram.to_term` as a `Functor` into it, reading `varname` when present, with
      `to_term(*names)` overriding the free-variable names.
- [ ] 8. `closed.Diagram.to_term`: the non-linear case, `Copy` for a repeated variable and `Discard`
      for an unused one.
- [ ] 9. Tests for T1–T5, including nested-binder and non-linear cases.
- [ ] 10. `CHANGELOG.md` entry under `[Unreleased]`, and a docstring example on `to_term`.
- [ ] 11. `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.

## The `repr` sub-decision is ruled

USER, [2026-08-14](https://github.com/discopy/discopy/pull/540#issuecomment-5291450910), verbatim:

> - `varname` should be an optional argument of `Wire.__init__`, it should appear in the `__repr__`
>   but not the `__str__`, it should not be included in the `__eq__`

So a diagram round-trips through `repr` carrying its names, `str` stays as a mathematician would
write it, and equality stays blind to the annotation. `__hash__` follows `__eq__` — it currently
hashes `(type, name, dom, cod)` and must keep doing exactly that, or `==` objects would hash apart.

The same comment asks for a rename that is **not** part of this branch, see below.

## Dependencies

- **#545 before this.** #541 blocks the discard case; #544 makes the roundtrip unstatable, since
  `t.eval().to_term().eval() == t.eval()` means nothing while `eval` does not preserve `cod`.
- **#442 before point 6**, or this branch writes its own substitution: composition in the term
  category is substitution, and `closed.Substitution` is broken on `main`. #442 fixes it with
  capture-avoidance.

## Scope

Roundtrip only. #375 bundled five unrelated fixes; `discard_factory` was already a `Discard` class
on `main` and `Substitution` belongs to #442. Bugs hit on the way get filed, not fixed here.

## Split out, not done here

The other half of USER's 2026-08-14 comment — renaming every `monoidal.Wire` subclass from `Ob` to
`Wire` — is a library-wide mechanical rename with nothing to do with the roundtrip. It is its own
branch and PR. The two do not collide: the rename touches `rigid`, `braided`, `biclosed`, `pivotal`,
`frobenius`, `feedback` and `quantum.circuit`, while `varname` touches only `monoidal.Wire`, which is
already named `Wire`.

## Knock-on

#376 is stacked on the abandoned #375 and names its terms "from the `varname` port annotations, with
fresh names generated from a global counter". Under B the annotations survive but the counter does
not. It needs re-targeting onto `main` and its naming redone against point 4 — not this branch's
work, and it should not merge before that.
