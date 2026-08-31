# TODO

Review round on [#443](https://github.com/discopy/discopy/pull/443), daydream6728's feedback approved by USER (toumix):

> i think its fine to define it like this by a triplet to stay close to the theory but the rest of discopy generally uses inheritance rather than composition, i.e. `class Monad(EndoFunctor)`. This way it would also inherit from functor axioms once property testing lands. could `Monad` (and its dependencies `Functor` and `EndoFunctor`) be defined in `abc`? what do you think @toumix ?

toumix, 2026-08-31: "yess makes a lot of sense!"

> why do we need this alias instead of calling Monad.from_maps directly? [on `make_monad`]

toumix, 2026-08-31: "good catch this is clearly slop"

- [WIP] @daylight-2026-08-31T11:25 `class Monad(EndoFunctor)` via inheritance instead of composition: drop
  the separate `self.functor`, fold `from_maps` into `__init__`, update every `monad.functor(...)` call site
  to `monad(...)` directly across `additive.py`/`channel.py`/`multiplicative.py`/`monad.py`/`test/kleisli.py`
- [ ] Remove `make_monad`, its one call site (`Monad.from_maps`) becomes `Monad(...)` directly at each of
  `Maybe`/`Powerset`/`Subdistribution`/`make_state`
- [ ] Smaller typing asks: `dist` -> PEP 695 `type Dist[X] = frozenset[tuple[X, float]]`, precise `Callable`
  signatures on `iterate_powerset`/`iterate_subdistribution` (correcting the two mismatched types/missing
  `tolerance` param in the reviewer's suggested diffs, which don't match the actual signatures)

**Deferred, not in scope here**: moving `Functor`/`EndoFunctor`/`Monad` into `discopy/abc.py` is a separate,
much wider architectural move touching imports across the whole codebase -- flagged back on the PR for
explicit scope before attempting it. Also deferred: `monad.py:416`'s "state as its own class" idea, which is
itself conditional on that wider move landing first.
