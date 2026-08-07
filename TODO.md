# Term-to-diagram roundtrip

> Term-to-diagram roundtrip https://github.com/discopy/discopy/pull/375
>
> the original PR was crap start again from scratch

Closes #372. Supersedes #375, which is abandoned: this branch starts from `main` and shares
no commits with it.

## What is actually true today

Measured on `ed4c0b3`, not assumed:

- There is no `to_term` anywhere in `discopy/`. The only occurrence of `varname` is a local
  variable in `biclosed.Ty.__call__`.
- `eval` quotients by alpha. `X(lambda y, left=True: y(f, left=True)).eval()` and the same term
  with `y` renamed to `z` are `==`, while the two terms are not. Renaming a *free* variable does
  change the diagram, but only because the term itself changed; the diagram holds types, never
  names.

So a term's variable names are not recoverable from its diagram, and no amount of care in
`to_term` changes that. #372's proposal is to put them there.

## The design decision this branch takes

#372 asks for a `varname` attribute on the `Ob` of a variable wire and on the curried wire, with
a global counter for fresh names. Taken literally that mechanism has two problems:

1. If `varname` enters `__eq__`, then `⟦λx.t⟧ != ⟦λy.t[y/x]⟧` and the free biclosed category is
   no longer alpha-quotiented. Currying stops being a function on morphisms.
2. A global counter makes `to_term` impure and order-dependent: calling it twice gives different
   names. STYLE.md rules that out ("data structures should not depend on sources of
   non-determinism").

This branch keeps #372's *intent* — a roundtrip that is faithful on the nose — and drops its
mechanism. Names stay out of the diagram; `to_term` picks the canonical alpha-representative
deterministically, from binder position rather than a counter. `Ty` and `Ob` are untouched.

**This reverses the letter of #372 and is USER's call to confirm.** The alternative, kept on the
table, is to store `varname` but exclude it from `__eq__`/`__hash__`, which buys an on-the-nose
roundtrip in both directions at the cost of a field equality ignores.

### The statements to be tested

With `to_term` partial, total on the image of `eval`:

- **T1** — for every term `t`: `t.eval().to_term().eval() == t.eval()`. Total, no naming caveat.
  This is the honest "faithful on the nose".
- **T2** — for every term `t`: `t.eval().to_term() == t` when `t` is already alpha-canonical, and
  equal up to alpha otherwise.
- **T3** — `to_term` raises a clear error off the image of `eval`, rather than returning nonsense.

### Terms are already a category, so `to_term` should be a functor

`TermBase` subclasses `Box` and already carries `dom` (the tensor of its free-variable types) and
`cod` (its type). That is exactly the shape of a morphism `context -> type`. Making that structure
explicit gives `to_term` for free through the existing `Functor` machinery:

| categorical structure | term |
|---|---|
| `id(A)` | `Variable` of type `A`, canonically named |
| `f >> g` | substitution |
| `f @ g` | context concatenation, renaming apart |
| `ev` | `Application` |
| `curry` | `Abstraction` |

`to_term` is then `Functor(ob_map=id, ar_map=Constant, cod=TermCategory)(diagram)`. `Functor` already
walks layers, so the layer-decomposition that made #375 sprawl is not written by hand at all. This
is the same move as `discopy.drawing`, where the layout algorithm is itself a functor.

**Dependency:** composition is substitution, and `closed.Substitution` is broken on `main` (it
recurses forever under abstractions and returns `None` on constants). #442 fixes it, with
capture-avoidance. Either #442 lands first or this branch needs its own substitution — decide
before starting point 4.

## Scope

Roundtrip only. #375 bundled five unrelated fixes; two of them are already on `main`
(`discard_factory` is a `Discard` class, not a lambda) or belong to #442 (`Substitution`). Every
bug hit on the way gets filed, not fixed here.

## Points

- [ ] 1. Confirm with USER the naming decision above (canonical vs. equality-invisible `varname`)
      before writing code — the whole shape of the branch depends on it.
- [x] 2. File the bugs found while measuring, as issues, and link them here: #541,
      `closed.Abstraction.eval` crashes with `ValueError: ... is not in list` on a constant
      function such as `X(lambda x: (X >> Y)('h'))`. `__check_dom__` admits zero occurrences of the
      bound variable but `eval` calls `.index()` unconditionally, so the discard path is
      unreachable and that case cannot be round-tripped.
- [ ] 3. Define the canonical naming scheme (de Bruijn level: free variables numbered by their
      index in `dom`, binders continuing the numbering) and test that it is a pure function of the
      diagram.
- [ ] 4. Make terms-in-context a `BiclosedCategory` instance: `id`, `then`, `tensor`, `ev`, `curry`
      per the table above.
- [ ] 5. `biclosed.Diagram.to_term` as a `Functor` into that category, with `to_term(*names)`
      taking optional free-variable names.
- [ ] 6. `closed.Diagram.to_term`: the same for the non-linear case, `Copy` for a repeated variable
      and `Discard` for an unused one. Blocked on #541.
- [ ] 7. Tests for T1, T2, T3, including the nested-binder and non-linear cases.
- [ ] 8. `CHANGELOG.md` entry under `[Unreleased]`, and a docstring example on `to_term`.
- [ ] 9. `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.

## Knock-on

#376 is stacked on #375 and names its extracted terms "from the `varname` port annotations, with
fresh names generated from a global counter". It inherits both problems above. Once the naming
decision is settled it needs re-targeting onto `main` and its naming redone to match — not this
branch's work, but it should not be merged before the decision.
