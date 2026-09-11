# TODO

Review feedback from @toumix on `discopy/axioms.py`, quoted verbatim:

> I'm not convinced.
>
> > The 3 the other way are Grid, ComposablePair and ComposableTriple: they generate the argument shapes a law quantifies over — two composable arrows, a pasting diagram — and state no law of their own.
>
> They should state laws of their own, e.g. the law of a composable pair ... is that its first element composes with the second.
>
> > Of the 92, most are ones we do expect to become testable eventually (Hypergraph, CMap, Tensor, Matrix, Stream, every Functor). Two kinds we don't:
>
> - an abstract base class can add more abstract methods I don't see the issue here
> - you've basically explained why these 92 should be `Testable`, they're just not implemented as such yet.

and the human's decision on where the merge lands:

> i want it inside 744

`Theory` and `Testable` become one class. `strategy` stops being an
`@abstractmethod` — which would make 66 concrete classes uninstantiable —
for a default that raises, so a class opts out of the matrix by not
generating its terms rather than by hiding its laws. The opt-out moves
from `axioms` to `strategy`, which is what is actually missing, and
shrinks: the abstract roots inherit the raising default instead of
declaring it, so `abc.Serialisable`, `abc.Category`, `cat.Ob` and
`cat.Arrow` all stop declaring anything.

- [ ] `Theory` absorbs `Testable`, with `strategy` raising by default
- [ ] `no_strategy` opts out of `strategy`, on the eight classes that
      inherit one they should not; `declared_axioms` goes
- [ ] `ComposablePair` and `Grid` state the laws they enforce in `__new__`
- [ ] `proptest/test_axioms.py` reads its carriers off `strategy`
- [ ] `CHANGELOG.md`, `CONTRIBUTING.md`, `AGENTS.md`
- [ ] `pflake8`, the test suite and the property matrix
