# TODO

> analyze the diff of this PR locally and come up with a way to factor as much
> as possible of the code via python's inheritance. simplify the code as much
> as possible, include the strict minimum to make test_properties.py work, and
> finding a way to get enough coverage in the unit test suite for strategies.
> every syntax module should get a single new test_strategy test in their
> corresponding test file. every generators in discopy.testing should get
> exactly one test, for ensuring that is succeeds on valid arguments, fails on
> invalid ones, and has a good enough search strategy that is able to find
> every type of box we expect.
> then, merge with main and report conflict resolution, and figure out whether
> new changes from main in the recent changelog items could further simplify
> the architecture.
> synthesize all your findings in a plan for a large refactoring that should
> aim to decrease the size of the diff as much as possible or organize the
> tests cleanly and uniformly.
> be very strict about boilerplate, uniformity is key.
> identify potential name clashes between axiom names and normal method names,
> using a consistent nomenclature.
> for example, instead of repeating the equations when overriding an axiom, we
> can introduce an `.up_to(f)` method on Equations that returns a new equation
> that rebinds its up_to argument with the provided function. then, we can
> weaken axioms by doing
> @axiom
> def my_axiom(cls, args...) -> Equation[C1]:
>    return super().my_axiom(args...).up_to(cls.normal_form)
>    # or
>    return AxiomError(super().my_axiom(args...))
> this is just an example, but try to resolve boilerplate by reusable
> abstrtactions in this fashion.
> work locally and finalize this PR.

- [WIP] Merge main into the branch, resolve conflicts, report the resolution.
- [WIP] Factor axiom overrides through inheritance: `Equation.modulo`,
  `AxiomError(super().axiom(...))`, `inapplicable(reason)`.
- [WIP] Consistent nomenclature: functor laws `preserves_*`,
  `transpose_axiom` -> `pivotality`; report remaining clashes.
- [WIP] One `test_strategy` per syntax module, one test per generator in
  `discopy.testing`; delete `proptest/test_strategies.py`.
- [WIP] Green: pflake8, unit suite with coverage, proptest.
- [WIP] Update CHANGELOG, write the refactoring plan, delete this file.
