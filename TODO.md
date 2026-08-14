# Rename every `monoidal.Wire` subclass from `Ob` to `Wire`

> - `class Ob(monoidal.Wire)` should go to `class Wire(monoidal.Wire)` same for braided and circuit
>   and everything that subclasses monoidal.Wire in general, we forgot them in the last rename
> - `varname` should be an optional argument of `Wire.__init__`, it should appear in the `__repr__`
>   but not the `__str__`, it should not be included in the `__eq__`

Split from [#540](https://github.com/discopy/discopy/pull/540), where USER left both bullets. The
second is the roundtrip's business and stays there; this branch is only the rename. They do not
collide — this touches the subclasses, `varname` touches `monoidal.Wire`, which is already named
`Wire`.

## Scope, measured

Seven classes subclass `monoidal.Wire`, directly or transitively:

| module | line | current |
|---|---|---|
| `discopy/rigid.py` | 166 | `class Ob(monoidal.Wire)` |
| `discopy/braided.py` | 70 | `class Ob(monoidal.Wire)` |
| `discopy/biclosed.py` | 182 | `class Ob(monoidal.Wire)` |
| `discopy/pivotal.py` | 62 | `class Ob(rigid.Ob)` |
| `discopy/frobenius.py` | 71 | `class Ob(pivotal.Ob)` |
| `discopy/feedback.py` | 159 | `class Ob(braided.Ob)` |
| `discopy/quantum/circuit.py` | 80 | `class Ob(frobenius.Ob)` |

**`cat.Ob` is not one of them** and keeps its name: it is the root object class, not a wire.
`monoidal.Wire` subclasses `cat.Ob`, so the boundary is exactly one level down from `cat`.

Roughly 58 bare `Ob` occurrences inside those seven modules, plus 12 qualified references elsewhere
(`rigid.Ob` 4, `frobenius.Ob` 6, `braided.Ob` 1, `pivotal.Ob` 1) and hits in `test/rigid.py`,
`test/tensor.py`, `test/hypergraph.py`, `test/utils.py`.

## Deprecate `Ob`, do not just drop it

USER, [2026-08-14](https://github.com/discopy/discopy/pull/566#discussion_r3785260040), verbatim:

> also add a DeprecationWarning when the user tries to construct an Ob like we did for
> `drawing.Equation`

This replaces the earlier plan of a clean break. The `drawing.Equation` precedent is a module-level
`__getattr__` in `discopy/drawing/__init__.py`, which fires only when the name is not found normally
— exactly the case once `Ob` is gone:

```python
def __getattr__(name):
    if name == "Ob":
        import warnings
        warnings.warn(
            "discopy.rigid.Ob is deprecated, use discopy.rigid.Wire instead.",
            DeprecationWarning, stacklevel=2)
        return Wire
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

One per renamed module. It covers construction, `from ... import Ob`, and `isinstance` alike, since
all three go through attribute access.

## The shim also keeps old serialisations loading

Class names are baked into both `repr` and `to_tree`:

```python
>>> Ob('x').to_tree()
{'factory': 'rigid.Ob', 'name': 'x'}
```

The earlier draft of this plan said trees serialized before the rename would stop loading. **That was
wrong, given the shim.** `utils.from_tree` resolves a factory string with

```python
return getattr(module, factory).from_tree(tree)
```

and `getattr` consults the module `__getattr__`, so `'rigid.Ob'` resolves to `Wire` with a
`DeprecationWarning`. The class's own `from_tree` ignores the `factory` field — checked:
`Wire.from_tree({'factory': 'rigid.Ob', 'name': 'x'})` gives `cat.Ob('x')`. So old and new trees both
load, and the deprecation is a real deprecation rather than a rename with a warning bolted on.

## Points

- [WIP] @evening-bptwxh-2026-08-14 20:05 1. Rename the seven classes and every in-module reference: `ob = Ob` factory assignments, type
      hints, `__all__`/autosummary entries, docstrings.
- [WIP] @evening-bptwxh-2026-08-14 20:05 2. Update the 12 qualified references outside those modules, and the four test files.
- [WIP] @evening-bptwxh-2026-08-14 20:05 3. A module-level `__getattr__` in each of the seven modules, warning and returning `Wire`,
      following `discopy/drawing/__init__.py`. Check none of them already defines `__getattr__`.
- [WIP] @evening-bptwxh-2026-08-14 20:05 4. Test the deprecation: `Ob` still constructs, warns once with `DeprecationWarning`, and is
      the same class as `Wire`; and `from ... import Ob` warns too.
- [WIP] @evening-bptwxh-2026-08-14 20:05 5. Check `to_tree`/`from_tree` round-trips for each renamed class, **and** that a tree
      serialized with the old `'<module>.Ob'` factory string still loads through the shim.
- [WIP] @evening-bptwxh-2026-08-14 20:05 6. Check `eval(repr(x)) == x` still holds for each renamed class — `repr` goes through
      `factory_name`, so it should follow automatically, but it is the invariant `STYLE.md` names.
- [WIP] @evening-bptwxh-2026-08-14 20:05 7. Grep the docs for `.Ob` references that are now stale, including `docs/_api` autosummary
      stubs.
- [WIP] @evening-bptwxh-2026-08-14 20:05 8. `CHANGELOG.md` entry under `[Unreleased]`, in `### Changed`, noting the missing half of the
      earlier rename and the deprecation.
- [WIP] @evening-bptwxh-2026-08-14 20:05 9. `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.
