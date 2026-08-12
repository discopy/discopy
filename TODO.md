# TODO

Prompt ([#558](https://github.com/discopy/discopy/issues/558), verbatim):

> We need the construct $Para(C)$ for a symmetric monoidal category $C$
>
> ```python
> class Para[C0, C1](NamedGeneric["category"], SymmetricCategory[C0, C1]):
>   dom: C0
>   cod: C0
>   mem: C0
>   inside: C1
>   def __post_init__(self):
>     assert self.inside.dom == self.dom @ self.mem
>     assert self.inside.cod == self.cod
> ```
>
> There's already an initial implementation there
> https://github.com/discopy/discopy/pull/325 but we want to refactor it with
> the new `abc`.

USER rulings (Daylight session, 2026-08-12): "param not mem", "reparam in
scope", "do the work yourself don't wait for evening".

- [WIP] @eloquent-pasteur-2026-08-12 13:36 `discopy/para.py`: `Para(SymmetricCategory, NamedGeneric['category'])`
  with fields `dom, cod, param, inside`, methods `id`, `then`, `tensor`,
  `swap`, `trace` and `reparam`
- [WIP] @eloquent-pasteur-2026-08-12 13:36 `Reparam`, the 2-cells of reparametrization, with identity, vertical
  (`>>`) and horizontal (`@`) composition
- [WIP] @eloquent-pasteur-2026-08-12 13:36 doctests: the axioms, drawings for `then` and `tensor`, an example over
  `python.Function`
- [WIP] @eloquent-pasteur-2026-08-12 13:36 `test/test_para.py`
- [WIP] @eloquent-pasteur-2026-08-12 13:36 wire into `discopy/__init__.py`, `docs/api/semantics.rst`,
  `docs/discopy.bib` and `CHANGELOG.md`
- [WIP] @eloquent-pasteur-2026-08-12 13:36 `uv run pflake8 discopy` and `uv run pytest --skip-extra` green
