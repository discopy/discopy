# TODO

> i have an idea to be able to avoid re-defining trivial subclasses for generators, and instead define them once:
>
> ```
> # cat.py, defines box_factory once:
> class Arrow(FreeCategory):
>     @cached_property
>     @classmethod
>     def generator_factory(cls) -> type[Box]:
>         generator_bases = base.generator_factory for base in cls.__bases__ if issubclass(base, Arrow)
>         return type("Box", (*generator_factory, cls), {})
>
> class Box(cat.Box, Diagram):
>    pass
>
> # symmetric.py, defines swap_factory once, no need to define Box again
> class Diagram(monoidal.Diagram, SymmetricCategory):
>     @cached_property
>     @classmethod
>     def swap_factory(cls) -> type[Swap]:
>         swap_bases = supcls.swap_factory for supcls in cls.__bases__ if issubclass(supcls, Diagram)
>         return type("Swap", (*swap_bases, cls), {})
>
> class Swap(Box):
>     ...
>
> # closed.py
> # nothing to do here anymore or any other symmetric syntax module, no need to define closed.Swap, no need to assign closed.Diagram.swap_factory = closed.Swap
> # instead, just define Swap = Diagram.swap_factory instead of Diagram.swap_factory = Swap
> ```
>
> perform this refactor globally and simplify discopy as much as possible in the process.

- [ ] `utils.Factory`, a class attribute building the generators of a category on demand from those of its bases, installed by `utils.factory` for every generator a decorated class inherits, with `utils.generators` listing them; doctests and unit tests.
- [ ] `cat.Arrow` type-checks its boxes itself so that `monoidal.Diagram.generator_factory` can be `monoidal.Box` rather than `cat.Box`.
- [ ] Replace every trivial generator subclass of `discopy/*.py` by `Name = Diagram.name_factory`, keeping the ones with behaviour assigned right after their definition; export the generators each level now builds (`Sum`, `Bubble`, `Trace`); `compact.Diagram.trace_factory` as a classmethod; `closed.Diagram.is_linear` without per-box flags.
- [ ] Same for `discopy/grammar` and `discopy/quantum`.
- [ ] README: the cooking example and the theory bullet on `swap_factory`.
- [ ] Tests: `test/utils.py` for the mechanism, a test that every generator a library module builds is exported by name, existing tests adjusted.
- [ ] `CHANGELOG.md` entry; `uv run pflake8 discopy`, `uv run coverage run -m pytest` and the property tests green.
