# TODO

Issue [#624](https://github.com/discopy/discopy/issues/624), verbatim:

> `CMap.to_hypergraph` recomputes `ports` per box: 226 s for 3200 boxes
>
> `Hypergraph.from_map` (`hypergraph.py:850`, reached by `CMap.to_hypergraph`) calls `CMap.ports`
> (`cmap.py:295`) once per box, and `ports` rebuilds the whole port list each time; with
> `Ty.__getitem__` and `Ty.__init__` inside that loop the cost is quadratic with a large constant.
>
> Measured on main (`8761d40f`), same chain of `n` boxes as in the `Hypergraph.from_diagram` issue:
>
> | n | `CMap.from_diagram` | `CMap.to_hypergraph` |
> |---|---|---|
> | 50 | 7 ms | 17 ms |
> | 200 | 25 ms | 246 ms |
> | 800 | 103 ms | 5.4 s |
> | 3200 | 480 ms | 226 s |
>
> cProfile at n=300: `cmap.py:295(ports)` 1503 calls / 2.07 s cumulative out of 2.07 s total; under
> it `monoidal.py:356(__iter__)` 375k calls, `cat.py:229(__getitem__)` 190k calls,
> `monoidal.py:272(__init__)` 193k calls. Caching `ports` once (it is a `cached_property`
> candidate — `CMap` is immutable) or building the wiring from `edges` directly would make this
> linear.

toumix, on the issue: *"good catch let's fix it!"*

- [x] reproduce the quadratic blow-up against the pre-fix commit (`8761d40f`) to confirm the issue
      is real: n=800 `to_hypergraph` took 14.9 s there
- [x] confirm `CMap.ports` is already a `functools.cached_property` on current `main` (landed as
      part of #532's unrelated `CMap`/`Hypergraph` alignment refactor, merged after this issue was
      filed) and that `CMap` has no attribute mutation after `__init__`, so the cache is sound
- [x] confirm the fix on current `main`: n=800 `to_hypergraph` is 65.6 ms, n=3200 is 294.9 ms —
      linear, not the 226 s reported
- [x] add a regression test in `test/cmap.py` locking in that `ports` is cached (identity across
      repeated access) so a future refactor can't silently turn it back into a plain `@property`
- [x] add a `### Performance` entry to `CHANGELOG.md`'s `[Unreleased]` section for #624
- [x] `uv run pflake8 discopy` and `uv run coverage run -m pytest` green
