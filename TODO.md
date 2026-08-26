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

> - rename preserves_tensor to monoidal, preserves_swap as symmetric,
>   preserves_copy as markov, etc...
> - define modulo on axioms too, allowing
>   bifunctoriality = BaseClass.bifunctoriality.modulo(normal_form)
> - define failing(self, reason: str) -> Self which wraps the inner function
>   to make it return an axiom error with the error message and the
>   constructed equation
> - similarly, define inapplicable as a method of Axiom and do every
>   override in one statement.
> - avoid all boilerplate related to parametrized classes in the property
>   test suite and discopy.testing. move all the axiom logic to dynamic
>   dispatch in the relevant classes

- [x] Merge main into the branch, resolve conflicts, report the resolution.
- [x] Factor axiom overrides through inheritance: `Axiom.modulo`,
  `Axiom.failing`, `Axiom.inapplicable`, every override one statement.
- [x] Consistent nomenclature: functor laws named after their level
  (`monoidal`, `braided`, ..., `rigid_cups`/`rigid_caps`),
  `transpose_axiom` -> `pivotality`; report remaining clashes.
- [x] One `test_strategy` per syntax module, one test per generator in
  `discopy.testing`; delete `proptest/test_strategies.py`.
- [x] Flatten the property matrix to one parametrized test over `CARRIERS`,
  argument generation on `Axiom.strategy()`; delete `proptest/strategies.py`.
- [x] Green: pflake8, unit suite with coverage, proptest.
- [x] Update CHANGELOG, write the refactoring report, delete this file.
