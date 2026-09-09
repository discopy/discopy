# TODO

> implement this issue: https://github.com/discopy/discopy/issues/710
>
> * abc.py: TracedCategory should inherit from FeedbackCategory
> * traced.Diagram should inherit from feedback.Diagram
> * traced.Ty should inherit from feedback.Ty with delay as identity
>
> while you're at it, try to generalise the implementation of traced categories and upstream as much as possible into this newfound parent class.

- [x] Move `abc.FeedbackCategory` down the hierarchy from `MarkovCategory` to `MonoidalCategory` and make `abc.TracedCategory` inherit from it, with `delay` the identity and `feedback` given by the trace.
- [x] Give traced types the trivial delay concretely: `monoidal.Ty.delay` is the identity like `unwind`, overridden by `feedback.Ty`; document the new methods in `traced`. The free `feedback.Diagram` and `feedback.Ty` cannot conversely become base classes of `traced.Diagram` and `traced.Ty`: they are already their subclasses through `markov`, so the concrete inheritance goes the other way around and the shared interface lives in `abc`.
- [x] Replace the monkey-patching in the `feedback` module note by the now built-in methods, and declare `stream.Stream` the `FeedbackCategory` it already implements.
- [x] Add tests and a changelog entry, run `pflake8` and the test suite.
