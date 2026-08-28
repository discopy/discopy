# TODO.md

Review round from @daydream6728 on
[#658](https://github.com/discopy/discopy/pull/658), quoted verbatim.

> here are some minor comments. about @toumix-agents's suggestion: i tried
> extracting the search strategies implementations before, and was in fact
> the original design.
> First address these comments for now, I will come up with a plan to try
> again to decouple the search strategies implementations from the discopy/
> tree.

- `discopy/abc.py:104` — "replace abc.Equation by a parametric
  (NamedGeneric) cat.Equation and define cat.Equation as
  abc.Equation[Arrow]. please avoid cyclic dependencies and local imports."
- `discopy/testing.py:51` — "do we really need the quotation marks here?
  isnt the evaluation of typing annotations deferred nowadays? for which
  python version would it break to just write `st.SearchStrategy[T]` here
  and on every other `strategy` method?"
- `discopy/testing.py:154` — "make it so that we don't need this function
  at all. CMap should implement is_boundary_connected too and this function
  should disappear."
- `discopy/testing.py:833` — "this only makes sense for a diagram strategy,
  make that clear from the type signature."
- `discopy/testing.py:846` — "useless, remove this function" / "same as
  below, delete this function"
- `discopy/testing.py:856` — "remove this function altogether by making
  axioms raise the axiom errors instead of returning them."
- `discopy/testing.py:872` — "delete this function, make
  abc.Category.axioms return a `dict[str, Axiom]` out of the box."
- `proptest/test_eq_hash.py:18` — "this module isn't testing much, remove
  it altogether.
  instead, just use specify the axioms modulo(hash) when the hash function
  of a data structure is known to be invariant under that equation."
- `proptest/test_drawing.py:20` — "make a proper non-test module in
  proptest/ to store this CARRIERS, we don't want test files to import each
  other"
- `proptest/test_counterexamples.py:26` — "why do we need this? are we
  relabeling every type to "a"? why?"
- `pyproject.toml:55` — "don't pin version in the pyproject file, let
  uv.lock dictate which version we use"
- `BUGS.md:1` — "delete this file across all levels of the PR stack"

## Work

- [x] Move the `Equation` implementation into `abc` as a `NamedGeneric`
      parameterised by its arrow type, with `cat.Equation` as its
      instantiation, so `Category.equation_factory` needs no local import.
- [x] Drop the quotes on the `st.SearchStrategy[T]` annotations, checking
      that nothing evaluates them at runtime.
- [x] Give `CMap` an `is_boundary_connected` and delete the helper.
- [x] State in `assert_strategy_finds`'s signature that it takes a diagram
      carrier.
- [x] Delete `assert_verdict`.
- [x] Make a broken axiom raise its `AxiomError` instead of returning it,
      and delete `holds`.
- [x] Delete `declared_axioms`, returning `dict[str, Axiom]` from
      `abc.Category.axioms`.
- [x] Delete `proptest/test_eq_hash.py`.
- [x] Move `CARRIERS` and its parametrisation into a non-test module of
      `proptest/`.
- [x] Answer the question on `proptest/test_counterexamples.py`, and drop
      the relabelling if it is not needed.
- [ ] Unpin `pytest-xdist` in `pyproject.toml`, regenerating `uv.lock`.
- [x] Delete `BUGS.md` and every reference to it.
