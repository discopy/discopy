# TODO

> we don't need the axiom/classaxiom distinction. instead, make every axiom a classmethod implicitly.
> we will never need to call an axiom as an instance method, simplify the code accordingly.

- [ ] One `Axiom`, a classmethod of its carrier implicitly: drop `ClassAxiom` and `classaxiom`, state the three laws of `Strategy` over a generated term, simplify `parameters`, `scope` and `__call__`
- [ ] Follow through in `abc.py`, `cat.py`, the tests, the docs, the changelog and the PR body; run `pflake8`, the test suite and the property matrix
