# TODO

> ok nevermind, the @Generator was a good idea, but i feel like it should just be named @factory. however there is already a utils.factory decorator in discopy, what is it used for? i have a hunch that we can remove it with recent changes including this one. can you try to see if removing the existing @factory altogether is possible?

- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Remove the class decorator `utils.factory` and the `factory` attribute: `Category.ar` is the class itself, `cat.Arrow.ar` the first class of the method resolution order that is an `Arrow` but neither a `Box` nor a `NamedGeneric` subscript, `Matrix.ar` the first that is not a subscript; every `@factory` decoration and `.factory` read goes, in the package, the tests, the benchmarks, the notebooks and the README.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Bring back the generator-declaring decorator under the name `factory`, in place of the explicit `cached_classproperty` methods.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Tests, CHANGELOG; `pflake8`, the full suite and the property tests green.
