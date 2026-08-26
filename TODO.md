Review round, USER (toumix), 2026-08-25:

> This should go to the code not the tests (`test/closed.py:9`, the `Unitype` class)
>
> move this to the code too (`test/closed.py:165`, the `RightmostFirst` strategy)
>
> This should be cod, find some other name for what you called called cod (e.g.
> head_cod would work) (`discopy/closed.py:478`, `BohmTree`)
>
> This breaks the DisCoPy STYLE.md in the sense that we cannot hope that
> `eval(str(X)) == X` anymore (`discopy/closed.py:412`, `Substitution.fresh`)
>
> use indices e.g. x -> x1 -> x2 etc. instead (`discopy/closed.py:412`, same)

- [ ] Move `Unitype` from `test/closed.py` into `discopy/closed.py`
- [ ] Move `RightmostFirst` from `test/closed.py` into `discopy/closed.py`
- [ ] Rename `BohmTree.cod` to `head_cod`, rename `BohmTree.ty()` to `BohmTree.cod()`
- [ ] `Substitution.fresh` appends numeric indices instead of primes, so a
      renamed variable stays a valid identifier and `eval(str(term)) == term`
      keeps holding
