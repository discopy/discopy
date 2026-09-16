# TODO

> no @generator, just implement it the way i described it in my original prompt:
>
> ```
> class Diagram(monoidal.Diagram, SymmetricCategory):
>     @cached_property
>     @classmethod
>     def swap_factory(cls) -> type[Swap]:
>         if cls is Diagram:
>             return Swap
>         swap_bases = supcls.swap_factory for supcls in cls.__bases__ if issubclass(supcls, Diagram)
>         return type("Swap", (*swap_bases, cls), {})
>
> class Swap(Box):
>     ...
> ```

- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 `utils.Generator` removed; `utils.cached_classproperty`, a `classproperty` cached once per factory, since `cached_property` over `classmethod` returns the descriptor on Python 3.12; every factory written out as in the sketch, with the module of the built class and the box of the level as its last base.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Tests, README and CHANGELOG follow; `pflake8`, the full suite and the property tests green.
