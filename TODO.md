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

- [x] `discopy/para.py`: `Para(SymmetricCategory, NamedGeneric['category'])`
  with fields `dom, cod, param, inside`, methods `id`, `then`, `tensor`,
  `swap`, `trace` and `reparam`
- [x] `Reparam`, the 2-cells of reparametrization, with identity, vertical
  (`>>`) and horizontal (`@`) composition
- [x] doctests: the axioms, drawings for `then` and `tensor`, an example over
  `python.Function`
- [x] `test/test_para.py`
- [x] wire into `discopy/__init__.py`, `docs/api/semantics.rst`,
  `docs/discopy.bib` and `CHANGELOG.md`
- [x] `uv run pflake8 discopy` and `uv run pytest --skip-extra` green

Round 2 (USER, Daylight session, 2026-08-12, verbatim): "it's nice you added
the trace method but it made me realise we should have a Para for each class
in the hierarchy below symmetric"

- [x] `Para.lift` and a subclass per abc class below `SymmetricCategory`:
  `Markov` (copy), `Closed` (ev, curry), `Feedback` (delay, feedback),
  `Compact` (cups, caps) and `Hypergraph` (spiders), each defaulting to the
  free category of its level
- [x] doctests and tests for each level, `CHANGELOG.md` extended
- [x] file the confusing `left` defaults of `ev`/`curry` across `abc`,
  `biclosed`, `closed` and `rigid` as an issue

Round 3 (USER, Daylight session, 2026-08-12, verbatim): "yes lift is the
injection functor from a category into its para! i was thinking of adding a
Para class in each module so the naming would be symmetric.Para, compact.Para
etc. but maybe your approach is cleaner, it's just weird to have para.Compact
but not para.Symmetric, let's rename para.Para to it"

- [x] rename `para.Para` to `para.Symmetric` everywhere, note in `lift`'s
  docstring that it is the injection functor

Round 4 (USER, review, 2026-08-12, verbatim): "this should go to its own
Traced subclass (don't get tripped up by the fact symmetric.Diagram is traced
by default)"

- [x] move `trace` from `para.Symmetric` to a dedicated `para.Traced`
  subclass, preserving the hierarchy even though the default free symmetric
  category happens to implement trace

Round 5 (USER, Daylight session, 2026-08-12): asked the cost of doing the
trace/symmetric split properly in the abc, pointing at the design USER
endorsed on issue #349 (2026-06-26, verbatim): "What we can do is drop it in
the abc (for balanced, symmetric and markov) and add it explicitly in the
corresponding diagram classes i.e. balanced.Diagram is still a subclass of
traced.Diagram."

- [x] drop `TracedCategory` from `BalancedCategory` in `abc.py` (closes
  #349); `para.Symmetric` declares `SymmetricCategory` again, `Markov` moves
  back onto `Symmetric`, and `Traced`'s bases flip to `(TracedCategory,
  Symmetric)` to keep the MRO consistent with `CompactCategory`

Round 6 (USER, review, 2026-08-13, verbatim on abc.py): "No this makes no
sense whatsoever just revert. The original definition was correct here." and
"This is what you were supposed to do: class SymmetricCategory[C0, C1](
BraidedCategory[C0, C1]):" — the split lives at symmetric, not balanced;
"move Balanced further down, just before RibbonCategory"; the twist "will
move further down when symmetric and balanced meet again, i.e. in compact".

- [WIP] @eloquent-pasteur-2026-08-13 07:33 revert `BalancedCategory(BraidedCategory, TracedCategory)`; make
  `SymmetricCategory` extend `BraidedCategory` directly; move `Balanced` just
  before `RibbonCategory`; `twist = id` moves to `CompactCategory`
- [WIP] @eloquent-pasteur-2026-08-13 07:33 keep only `FongEtAl19` in the bib, drop `Gavranovic24` and
  `CapucciEtAl22`, trim the docstring citation
- [WIP] @eloquent-pasteur-2026-08-13 07:33 drop the `not hasattr(Symmetric, "trace")` assert and the doctest-
  repeating half of `test_python`
