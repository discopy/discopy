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

## No compatibility alias

The previous rename, `monoidal.Ob` to `monoidal.Wire`, left no `Ob = Wire` alias behind. This one
follows that precedent: a clean break, nothing deprecated.

## Serialization changes, deliberately

Class names are baked into both `repr` and `to_tree`:

```python
>>> Ob('x').to_tree()
{'factory': 'rigid.Ob', 'name': 'x'}
```

After the rename that becomes `'rigid.Wire'`, so any tree serialized before it will not load. This
is the same cost the `monoidal.Ob` rename already paid, and it is why the `to_tree`/`from_tree`
round-trip tests are on the checklist rather than assumed.

## Points

- [ ] 1. Rename the seven classes and every in-module reference: `ob = Ob` factory assignments, type
      hints, `__all__`/autosummary entries, docstrings.
- [ ] 2. Update the 12 qualified references outside those modules, and the four test files.
- [ ] 3. Check the `from_tree` registry resolves the new `factory` strings, and that
      `to_tree`/`from_tree` round-trips still pass for each renamed class.
- [ ] 4. Check `eval(repr(x)) == x` still holds for each renamed class — `repr` goes through
      `factory_name`, so it should follow automatically, but it is the invariant `STYLE.md` names.
- [ ] 5. Grep the docs for `.Ob` references that are now stale, including `docs/_api` autosummary
      stubs.
- [ ] 6. `CHANGELOG.md` entry under `[Unreleased]`, in `### Changed`, noting the missing half of the
      earlier rename.
- [ ] 7. `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.
