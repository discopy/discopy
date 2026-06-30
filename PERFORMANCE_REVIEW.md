# DisCoPy — Performance Bottleneck Review

**Scope:** `discopy/{hypergraph,matrix,tensor,cat,monoidal,utils,rigid,symmetric,frobenius}.py`
**Status:** Analysis only — no functional code changed by this report.

All line numbers below were **verified directly against the source** unless marked
`(unverified)`. Emphasis: **numerical representations** (`matrix.py`, `tensor.py`) and
**hypergraphs** (`hypergraph.py`).

---

## Global context & blockers (read first)

These apply to almost every fix below — listed once here so they aren't repeated.

- **No benchmarks exist.** `grep -ri "benchmark\|timeit\|perf_counter" test/ discopy/` returns
  nothing. There is no performance regression guard. **Before fixing anything, add a micro-benchmark**
  (e.g. build a depth-N diagram, compose N boxes, eval an N-box tensor net) so improvements are
  measurable and regressions caught. Correctness is currently guarded only by **doctests** (in every
  module) and the `test/` suite (`test/{syntax,semantics,quantum,grammar,drawing,utils}`).
- **Immutability is assumed but not enforced.** `Ty`, `Arrow`, `Diagram`, `Box`, `Hypergraph` are
  treated as value objects (they define `__hash__`/`__eq__` and are used as dict/set keys). This is
  what *permits* memoization — but also means any cache must key on identity/structure and never be
  invalidated by mutation. Confirm no code mutates `.inside`/`.array`/`.flat_wires` in place before
  caching derived data.
- **`__hash__`/`__eq__` semantics are load-bearing.** Several diagram subclasses opt into
  hypergraph equality (`use_hypergraph_equality`, e.g. symmetric.py:230-237). Don't "fix" a hash
  without checking which equality contract the class advertises, or you'll break set/dict dedup in
  `normal_form`, functor `ar` mappings, and the test suite's `assert d1 == d2`.
- **`NamedGeneric` parametrization.** Many classes are `Cls[dtype]`/`Cls[Category]` dynamic
  subclasses (utils.py:148-202). `type(self)`, `self.factory`, `self.category` lookups are
  everywhere in hot paths. A cache keyed on `cls` is safe; a cache keyed on instances must account
  for parametrized subclasses being distinct types.
- **Backend portability.** `matrix.py`/`tensor.py` must work across numpy / jax / pytorch /
  tensorflow / sympy `object` dtype. Any "just call `np.eye`" fix must go through the active backend
  (`with backend() as np:`) and must not assume in-place mutation (jax arrays are immutable;
  `tensor.py:189` already special-cases this incorrectly — see V5/T2.6).

---

## TIER 0 — Highest impact (every operation pays these)

### T0.1 — `cat.Arrow.__init__` re-validates the whole arrow on every composition
- **Where:** `cat.py:226-236`; triggered by `cat.py:311-328` (`then`).
- **Verified.** `then` ends with `return self.factory(inside, dom, cod)` — **no `_scan=False`** — so
  `__init__` (default `_scan=True`) re-walks the *entire accumulated* `inside`:
  ```python
  for box in inside: assert_isinstance(box, Box)
  for f, g in zip((Id(dom),) + inside, inside + (Id(cod),)): assert_iscomposable(f, g)
  ```
  Composing N boxes left-to-right → **O(N²)** validations + two throwaway tuple allocations and two
  throwaway `Id(...)` objects **per step**.
- **DESIGN NOTE — this is partly intentional, not a pure bug.** `_scan` is a *documented* parameter
  ("Whether to check composition", cat.py:180) and the codebase passes `_scan=False` everywhere it
  is provably safe: `Arrow.id` (309), `__getitem__` slices (246, 257), `subs` (389), `Box.__init__`
  (490), `Sum.from_tree` (457), `Diagram.tensor` (monoidal.py:588), `dagger` (rigid.py:328),
  `snake_removal` (rigid.py:498), `feedback.py:322`, `braided.py:104`. The reason `then` (cat.py:
  311-328) keeps `_scan=True` is **soundness**: `then` itself **never checks `self.cod ==
  other.dom`** — the `_scan` loop in `__init__` is the *only* composability check on the `then` path.
  So you must NOT naïvely flip it to `False`, or `f >> g` with mismatched types silently builds a
  broken arrow. It is, however, still *over*-checking: the bodies of `self.inside`/`other.inside`
  were already validated when those arrows were built; the only novel facts are the **K-1 joins**
  (`self.cod==others[0].dom`, `others[i].cod==others[i+1].dom`).
- **Fix (soundness-preserving):** Move the join check *into* `then` and then pass `_scan=False`:
  ```python
  def then(self, *others):
      if any(isinstance(other, Sum) for other in others):
          return self.sum_factory((self,)).then(*others)
      parts, dom, cod = [self.inside], self.dom, self.cod
      for other in others:
          assert_isinstance(other, self.factory)
          assert_isinstance(self, other.factory)
          assert_iscomposable(... cod ..., ... other.dom ...)   # check ONLY the new join
          parts.append(other.inside); cod = other.cod
      return self.factory(tuple(chain.from_iterable(parts)), dom, cod, _scan=False)
  ```
  This drops the per-`then` body rescan (O(N·K) → O(K) composability checks) **and** the
  `inside + other.inside` quadratic tuple concat (cat.py:327, O(N·K) → O(N)). It mirrors exactly what
  `Diagram.tensor` already does (tensor has no joins → no check needed; `then` has K-1 joins → check
  those). Separately apply T0.7 to drop the two `Id(...)` allocations from the remaining `_scan=True`
  public path.
- **Blockers:** `_scan` is the *only* place box-type and composability are checked; external callers
  rely on `Arrow(...)` validating hand-built `inside`, so keep `_scan=True` the public default. The
  `assert_iscomposable` signature takes two `Composable`s — adapt the call to compare `cod` vs
  `other.dom` (may need a tiny helper rather than fabricating dummy objects). Verify `test/syntax/`
  AxiomError tests still fire on mismatched composition.
- **Grep:** `_scan`, `def then`, `assert_iscomposable`, `Id(dom)`.

### T0.2 — `monoidal.Diagram.boxes` / `offsets` are recomputed `sum([...], [])` properties, hammered inside `normalize`
- **Where:** `monoidal.py:590-593` (`boxes`), `595-598` (`offsets`); abused at `monoidal.py:855-857`
  and `rigid.py:430,446,448` etc.
- **Verified.**
  ```python
  @property
  def boxes(self):   return sum([layer.boxes for layer in self.inside], [])  # O(L) layers, sum→O(L²)
  @property
  def offsets(self): return list(len(left) for left, _, _ in self)
  ```
  `normalize` (monoidal.py:855-857) reads `diagram.boxes[i]`, `diagram.boxes[i+1]`,
  `diagram.offsets[i]`, `diagram.offsets[i+1]` **inside the per-position loop** → rebuilds both whole
  lists 4× per iteration. With the quadratic `sum`, the inner loop of `normalize` is **≈O(L³)** on
  what should be O(L). `rigid.snake_removal.follow_wire` (rigid.py:430) indexes `diagram.boxes[i]`
  *inside its while loop* too (see T1.5 / Tier 4).
- **Fix:** (a) Make `boxes`/`offsets` `cached_property` (diagrams are immutable). (b) Even without
  caching, hoist `boxes = diagram.boxes; offsets = diagram.offsets` outside the `normalize` loop and
  index those. (c) Replace `sum([...], [])` with `list(chain.from_iterable(...))`.
- **Blockers:** `cached_property` needs `__dict__` (fine; these aren't `__slots__`). Confirm nothing
  mutates `inside` after construction (it shouldn't — `Diagram` is a value type).
- **Grep:** `def boxes`, `def offsets`, `sum(`, `\.boxes\[`, `\.offsets\[`, `cached_property`.

### T0.3 — `__hash__` via `hash(repr(self))` rebuilds the full structure string
- **Where:** `cat.py:282-283` (Arrow), `cat.py:541-542` (Box→`Arrow.__repr__`),
  `monoidal.py:176-177` (Ty), `monoidal.py:287-288` (PRO), `rigid.py:587-588` (rigid Box),
  `utils.py:759-768` (`Node`), `symmetric.py:233-237` (`__hash__` falls back to `hash(repr(self))`).
- **Verified** (all sites read).
  ```python
  def __hash__(self): return hash(repr(self))          # cat.Arrow / Ty / ...
  def __hash__(self): return hash(cat.Arrow.__repr__(self))  # rigid.Box
  def __hash__(self): return hash(repr(self))          # utils.Node
  ```
  Every dict/set membership, every functor `ar`-map lookup, every `normal_form` dedup pays O(total
  textual size) to hash one object, with **no caching**.
- **Fix:** `hash(self.inside)` for Arrow/Ty (inside is already a tuple of hashables), `hash((kind,
  frozenset(data.items())))` for `Node`. Optionally cache `self._hash` lazily.
- **Blockers:** Must stay **consistent with `__eq__`**. `Arrow.__eq__` compares `factory`,
  `is_parallel`, `inside` — so `hash((type(self).factory, self.dom, self.cod, self.inside))` is the
  safe key. For `Node`, `__eq__` compares `(kind, data)` — mirror that. For classes using
  `use_hypergraph_equality`, the hash must match hypergraph equality, **not** structural — leave
  those alone or hash the canonical hypergraph form (symmetric.py:226-228).
- **Grep:** `hash(repr`, `def __hash__`, `def __eq__`, `use_hypergraph_equality`.

### T0.4 — `utils.factory_name` is uncached, called on every type-check / repr / serialize
- **Where:** `utils.py:217-227`.
- **Verified.**
  ```python
  def factory_name(cls):
      module = cls.__module__.removeprefix('discopy.')
      return f"{module}.{cls.__name__}".removeprefix('builtins.')
  ```
  Deterministic in `cls`, but called by `assert_isinstance` (every constructor — see T0.5), every
  `__repr__`, every `to_tree`, and every error formatter. Probably the single most-called helper.
- **Fix:** `@functools.lru_cache(maxsize=None)` on `factory_name`, or store `cls._factory_name` once.
- **Blockers:** `lru_cache` holds a reference to every `cls` ever seen — fine (classes are
  long-lived), but note `NamedGeneric` creates *many* dynamic subclasses; cache grows with distinct
  parametrizations. Bounded in practice. No correctness risk (pure function of `cls`).
- **Grep:** `factory_name`, `removeprefix`.

### T0.5 — `utils.assert_isinstance` formats its error message on the success path
- **Where:** `utils.py:376-382`.
- **Verified.**
  ```python
  classes = cls if isinstance(cls, tuple) else (cls,)
  cls_name = ' | '.join(map(factory_name, classes))      # <-- ALWAYS runs (only used on failure)
  if not any(isinstance(object_, get_origin(cls)) for cls in classes):
      raise TypeError(messages.TYPE_ERROR.format(cls_name, factory_name(type(object_))))
  ```
  `cls_name` (which calls `factory_name` per class — T0.4) is computed **unconditionally**, even
  though it's only used to build the exception. `assert_isinstance` runs on essentially every
  box/diagram/Ty construction.
- **Fix:** Move `cls_name` and the second `factory_name` into the `if not ...:` branch. Hoist the
  one-tuple wrap; for the common single-class call, `isinstance(object_, get_origin(cls))` directly.
- **Blockers:** None — pure refactor. Doctests check the *message text* on failure (e.g.
  `test/.../*TypeError*`); keep the message format identical.
- **Grep:** `def assert_isinstance`, `cls_name`, `get_origin`.

### T0.6 — `matrix.backend()` is a contextmanager entered on *every* numerical op (and is global, not thread-safe)
- **Where:** `matrix.py:472-494`; `get_backend` (516-528) and `set_backend` (497-513) layer on top.
- **Verified.**
  ```python
  @contextmanager
  def backend(name=None, _stack=[config.DEFAULT_BACKEND], _cache=dict()):
      name = name or _stack[-1]; _stack.append(name)
      try:
          if name not in _cache: _cache[name] = BACKENDS[name]()
          yield _cache[name]
      finally: _stack.pop()
  ```
  Every `Matrix.__init__`, `then`, `tensor`, `dagger`, `map`, `round`, `__eq__`, `__repr__`, every
  `Tensor` op, and **every box inside `Functor.__call__`** pushes/pops the stack and does a dict
  membership test. `_stack`/`_cache` are **mutable default args** = process-global → also **not
  thread/async-safe**. `set_backend` reaches into `backend.__wrapped__.__defaults__[1][-1]` (brittle).
- **Fix:** Keep the public `backend(name)` contextmanager for *overrides*, but add a fast
  `get_backend()` that returns the cached singleton with no push/pop. Internally call
  `np = get_backend()` instead of `with backend() as np:` where no override scope is needed. For
  thread-safety, move `_stack` to a `contextvars.ContextVar`.
- **Blockers:** The `with backend('jax'):` override semantics (doctest at matrix.py:483-485) must be
  preserved — so the ContextVar must still be honored by `get_backend`. `set_backend`'s exact mutation
  is doctested (matrix.py:506-511); reimplement it to set the ContextVar default.
- **Grep:** `with backend()`, `def backend`, `def get_backend`, `def set_backend`, `_stack`, `_cache`.

### T0.7 — `Arrow.__init__` allocates `Id(dom)` and `Id(cod)` purely for the composability scan
- **Where:** `cat.py:235`.
- **Verified.** `zip((Id(dom),)+inside, inside+(Id(cod),))` builds two new tuples and two `Id`
  arrows (each a full `cls.factory((), dom, dom, _scan=False)`) on every validated construction.
- **Fix:**
  ```python
  if inside:
      if inside[0].dom != dom or inside[-1].cod != cod: raise AxiomError(...)
      for f, g in zip(inside, inside[1:]): assert_iscomposable(f, g)
  ```
- **Blockers:** Empty-`inside` (identity) case must still validate `dom == cod` — keep that branch.
- **Grep:** `Id(dom)`, `zip((Id`.

---

## TIER 1 — Hypergraph hot paths & algorithms (`hypergraph.py`)

`Hypergraph.__init__` runs on **every** `then`/`tensor`/`dagger`/`rotate`/`interchange`/transform —
so anything quadratic here is felt everywhere in the hypergraph layer.

### T1.1 — Quadratic relabelling in `__init__`
- **Where:** `hypergraph.py:203` (tuple concat), `213` and `218` (relabel).
- **Verified.**
  ```python
  flat_wires = dom_wires + sum([x + y for x, y in box_wires], ()) + cod_wires   # O(boxes²)
  relabeling  = sorted(connected_spiders, key=flat_wires.index)                 # .index → O(n²)
  self.flat_wires = tuple(relabeling.index(s) for s in flat_wires)              # .index → O(n²)
  ```
  Three independent O(n²) patterns in the constructor everyone calls.
- **Fix:**
  - L203: `tuple(chain(dom_wires, *(x + y for x, y in box_wires), cod_wires))`.
  - L213: one pass `first = {}` recording first occurrence, then `sorted(connected_spiders,
    key=first.__getitem__)`.
  - L218: `inv = {s: i for i, s in enumerate(relabeling)}; self.flat_wires = tuple(inv[s] for s in
    flat_wires)`.
- **Blockers:** Relabelling defines the **canonical spider numbering** that `__eq__`/`__hash__`
  (graph iso, T1.4) and `to_diagram` depend on. The first-occurrence order must be preserved exactly
  (currently `sorted(..., key=flat_wires.index)` = order of first appearance) or canonical forms
  shift and equality/serialization tests break. The dict-based version preserves it; verify against
  `test/` hypergraph doctests (the `spider_wires`/`ports` doctests at hypergraph.py:244-282).
- **Grep:** `flat_wires.index`, `relabeling`, `sum(\n.*box_wires`, `def __init__` in hypergraph.

### T1.2 — `spider_wires` and `ports` are recomputed properties, used repeatedly inside `__init__` and every is/make_* method
- **Where:** `hypergraph.py:233-260` (`spider_wires`), `262-293` (`ports`).
- **Verified.** `__init__` calls `self.ports` at L209 *and* indexes `self.ports[i]` inside the
  validation loop at L226-229 — **rebuilding the entire ports list every iteration**. `ports` itself
  uses the quadratic `sum([[...]], [])` at L286-290. `spider_wires` is O(n) but recomputed on every
  access by `is_bijective`/`is_monogamous`/`is_causal`/`make_*`/`scalar_spiders`.
- **Fix:** Memoize both as `cached_property` (hypergraphs are value objects). At minimum, hoist
  `ports = self.ports` to a local before the L224-229 loop. Replace `sum([[...]], [])` at L286 with
  `list(chain.from_iterable(...))`.
- **Blockers:** `ports`/`spider_wires` are documented public properties with doctests
  (hypergraph.py:244-282) — output must be identical. Caching assumes immutability; `interchange`
  builds a *new* Hypergraph rather than mutating, so this holds.
- **Grep:** `def spider_wires`, `def ports`, `self.ports[`, `sum(\[\[`.

### T1.3 — `simplify` is super-quadratic (recomputes `to_diagram()` in the inner loop, then recurses from scratch)
- **Where:** `hypergraph.py:524-542`.
- **Verified.**
  ```python
  for i in range(len(self.boxes)):
      for j in range(len(self.boxes)):
          result = self.interchange(i, j)
          if len(result.to_diagram()) < len(self.to_diagram()):   # self.to_diagram() recomputed each (i,j)
              return result.simplify()                            # restart from (0,0)
  ```
  `self.to_diagram()` (which calls `make_monogamous().make_causal()`, recursive) is rebuilt for every
  `(i,j)`; `result.to_diagram()` likewise; success recurses and restarts. Docstring claims
  "quadratic" but it is far worse.
- **Fix:** Hoist `base = len(self.to_diagram())` outside both loops. Iterate `j` over `i+1..` only
  (interchange is symmetric in effect). Consider an explicit worklist instead of full restart.
- **Blockers:** `interchange(i, j)` for non-adjacent `i,j` may itself be expensive (check
  semantics); the simplification must still reach a fixpoint equal to the current behavior — guard
  with the doctest at hypergraph.py:535.
- **Grep:** `def simplify`, `to_diagram()`, `def interchange` (hypergraph).

### T1.4 — `__eq__` runs NetworkX graph isomorphism; `__hash__` runs Weisfeiler–Lehman; neither cached
- **Where:** `hypergraph.py:549-553` (`__eq__`), `555-557` (`__hash__`, `(unverified — confirm the WL
  call)`).
- **Verified (`__eq__`).**
  ```python
  return self.is_parallel(other) and is_isomorphic(
      self.to_graph(), other.to_graph(), lambda x, y: x == y)
  ```
  General iso is worst-case superpolynomial; both sides build a full `to_graph()`. Catastrophic when
  Hypergraphs are dict/set keys or compared in tests.
- **Fix:** Fast-path: if `is_parallel` **and** canonical tuples `(boxes, wires, spider_types,
  offsets)` are equal → `True` without iso. Only fall back to iso when canonical forms differ. Cache
  `self._hash` lazily.
- **Blockers:** The library *intends* iso-equality (two hypergraphs equal up to spider renaming).
  After T1.1 the canonical relabelling already normalizes spider numbering, so canonical-equality is
  a sound fast-path **for parallel, same-structure** graphs — but two graphs that are isomorphic yet
  built in different box orders must still compare equal, so you cannot *replace* iso, only short-
  circuit the common case. Validate against any `assert h1 == h2` in `test/`.
- **Grep:** `is_isomorphic`, `to_graph`, `def __eq__`/`__hash__` (hypergraph), `weisfeiler`.

### T1.5 — `interchange` reallocates 4 lists + 3 tuples then triggers the quadratic `__init__`
- **Where:** `hypergraph.py:514-522`.
- **Verified.** `list(self.boxes)`, `list(self.offsets)`, `list(self.box_wires)`, three swaps, three
  `tuple(...)`, then `type(self)(...)` → full T1.1 relabel. Called O(n²) times by `simplify` (T1.3).
- **Fix:** Build the three swapped tuples by slicing directly; add a trusted constructor path that
  skips relabelling (interchange preserves canonical spider numbering).
- **Blockers:** Needs the "fast constructor" from T1.1 to exist; must prove interchange truly
  preserves canonical form (it permutes boxes, not spiders — should hold).
- **Grep:** `def interchange` (hypergraph), `box_wires`.

### T1.6 — `bijection`, `make_monogamous`, `make_bijective`, `make_causal`: slice-and-`.index`/recurse
- **Where:** `bijection` `597-603` (verified pattern); `make_monogamous` `766-803`; `make_bijective`
  `720-749`; `make_causal` `858-883` `(make_* line numbers per sub-agent — confirm before editing)`.
- **Pattern (verified in `bijection`):** `spider not in flat_wires[i+1:]` + `flat_wires[i+1:]
  .index(spider)` → slice + linear scan twice per index = **O(n²)** time and memory; the `make_*`
  methods rebuild a whole `Hypergraph` (T1.1 cost) and **recurse** per fix, some round-tripping
  through `to_diagram` (which itself calls `make_monogamous().make_causal()` — mutual recursion).
- **Fix:** Precompute `spider → [positions]` (`defaultdict(list)`) once and pair up; process **all**
  offending spiders in a single pass instead of recursing per-fix; separate "compute fixes" from
  "render to diagram" so the transform doesn't re-enter `to_diagram`.
- **Blockers:** These encode the categorical normal-form algorithms (monogamy/causality); the
  rewrite must produce the *same* resulting diagram. Heavily doctested — verify every doctest in the
  `make_*`/`is_*` methods and `test/` hypergraph cases. Re-confirm exact line numbers for the
  `make_*` bodies (only `bijection`/`__init__` were line-checked here).
- **Grep:** `flat_wires\[.*:\]`, `\.index(`, `def make_`, `def bijection`, `def to_diagram`.

### T1.7 — `to_diagram` does `scan.index(...)` per port per box
- **Where:** `hypergraph.py:~972, ~994` `(unverified line numbers — confirm)`.
- **Pattern:** maintaining wire positions via `scan.index(dom_wires[i])` inside the box loop = O(len
  scan) per port. `to_diagram` is on the critical path of `simplify`/`eval`/printing.
- **Fix:** maintain a `dict[spider, position]` updated on each swap/box application.
- **Blockers:** Confirm line numbers; `to_diagram` output ordering must be byte-identical (doctests
  print diagrams).
- **Grep:** `def to_diagram`, `scan.index`, `scan =`.

### T1.8 — `from_callable` monkeypatches `cls.category.ar.__call__` without `try/finally`
- **Where:** `hypergraph.py:~1041, ~1051` `(unverified line numbers — confirm)`.
- **Pattern:** `cls.category.ar.__call__ = apply ... del cls.category.ar.__call__`. If `func`
  raises, the class is left with a clobbered `__call__`; also not concurrency-safe.
- **Fix:** wrap in `try/finally`; better, route through a thread-local/contextvar rather than
  mutating the class.
- **Blockers:** Behavioral — affects correctness under exceptions/threads, not throughput. Low risk
  to fix. Confirm lines.
- **Grep:** `__call__ =`, `def from_callable`.

### T1.9 — Smaller hypergraph allocations
- `to_graph` builds a fresh `Node` per **port** even when many ports share a spider — cache one Node
  per spider (`hypergraph.py:~1099-1132`, *unverified lines*). **Grep:** `Node("spider"`.
- `spring_layout` calls `self.ports` **three times** in one expression (`~1161`, *unverified*).
  **Grep:** `spring_layout`, `self.ports`.
- `rebracket` builds `box.dom @ box.cod` just to call `len()` (`hypergraph.py:~302-310`, verified
  region) — use `len(box.dom)+len(box.cod)`. **Grep:** `box.dom @ box.cod`.

---

## TIER 2 — Numerical core (`matrix.py`, `tensor.py`)

### T2.1 — Double array conversion on every Matrix/Tensor construction
- **Where:** `matrix.py:145-154` (`__new__` does `np.array(array)` to read dtype) + `156-161`
  (`__init__` does `np.array(array, dtype=...).reshape(...)`); `tensor.py:601-617` (`Box.__new__` +
  `_get_data_dtype`) + `tensor.py:107-112` (`Tensor.__init__` reshapes **again**).
- **Verified.**
  ```python
  # matrix.__new__: _array = np.array(array); dtype = ...; re-enter __new__(cls[dtype], ...)
  # matrix.__init__: self.array = np.array(array, dtype=self.dtype).reshape((dom, cod))
  # tensor.__init__: super().__init__(array, product(dom.inside), product(cod.inside))  # reshape #1
  #                  self.array = self.array.reshape(dom.inside + cod.inside)            # reshape #2
  ```
  When dtype is unknown the input list is converted twice; every `Tensor` reshapes twice. Internal
  callers (`then`, `tensor`, `dagger`) already pass a correctly-shaped backend ndarray and still pay
  `np.array(...).reshape(...)`.
- **Fix:** Add a trusted internal constructor (or fast-path in `__init__`) that skips
  `np.array`/`reshape` when `hasattr(array, "shape")` and shape/dtype already match. Pass the dtype-
  probed `_array` from `__new__` into `__init__` instead of re-converting. In `Tensor.__init__`,
  construct at the final shape directly (skip the flat `(product, product)` reshape).
- **Blockers:** dtype dispatch (`cls[dtype]`) relies on `__new__` returning the right parametrized
  class; the fast path must still route dtype correctly. `reshape` on some backends returns a view,
  on others a copy — don't assume aliasing. jax arrays are immutable — never `array[...] =`.
  Doctests assert exact `Matrix[int64](...)` reprs — dtype inference must be unchanged.
- **Grep:** `np.array(`, `.reshape(`, `def __new__`, `def __init__`, `_get_data_dtype`,
  `product(dom.inside)`.

### T2.2 — `Box.array` recomputes `np.array(self.data).reshape(...)` on every read
- **Where:** `tensor.py:619-624`.
- **Verified.**
  ```python
  @property
  def array(self):
      if self.data is not None:
          with backend() as np:
              return np.array(self.data).reshape(self.dom.inside + self.cod.inside)
  ```
  `Functor.__call__` reads `self(box).array` once per box; `to_tn`/`to_quimb` read `box.eval().array`.
  Re-converts the same Python data every time, and enters `backend()` (T0.6) each read.
- **Fix:** `@functools.cached_property` (Box is immutable) — but see blocker.
- **Blockers:** Backend can change at runtime via `set_backend`/`with backend(...)`. If `array` is
  cached under numpy then read under jax, the cached numpy array is wrong. Either (a) key the cache
  on the active backend, or (b) cache only the dtype-normalized host array and let callers move it to
  the backend, or (c) cache and document that `Box.array` follows the construction-time backend.
  Check `test/semantics` jax/torch tests.
- **Grep:** `def array`, `self.data`, `cached_property`.

### T2.3 — `Functor.__call__` recomputes `dim(other.dom @ scan[:off])` repeatedly and re-enters `backend()` per box
- **Where:** `tensor.py:355-390`.
- **Verified.**
  ```python
  dim = lambda scan: len(self(scan))           # runs the Functor on a Ty just to get a length
  for box, off in zip(other.boxes, other.offsets):
      ... dim(other.dom @ scan[:off]) ...      # rebuilt several times per iteration
      with backend() as np: ... np.moveaxis ... # opened/closed up to 3× per box
  ```
  `other.boxes`/`other.offsets` are the quadratic properties (T0.2). `dim(...)` constructs a new
  `Dim` via `@` and dispatches through `Functor.__call__` only to take `len`. `backend()` is entered
  up to three times per box.
- **Fix:** `dim(scan) == len(scan)` for a `Dim` (the `isinstance(other, Dim): return other` branch is
  identity) → replace `dim` with `len`. Maintain a **running integer offset** as `scan` updates
  instead of recomputing `dim(other.dom @ scan[:off])`. Hoist a single `np = get_backend()` (or one
  `with backend()`) around the whole loop. Hoist `other.boxes`/`other.offsets` to locals (or fix
  T0.2).
- **Blockers:** Must preserve `moveaxis`/`tensordot` index math exactly. The `Swap` branch
  (tensor.py:366-377) and the general branch compute different axis maps — verify with
  `test/semantics` tensor doctests (e.g. tensor.py:400-403, 418-420).
- **Grep:** `def __call__` (tensor Functor), `dim = lambda`, `np.moveaxis`, `np.tensordot`,
  `scan[:off]`.

### T2.4 — `Tensor.tensor`/`dagger`/`swap`/`conjugate`: loop-invariant `len(dim @ dim @ dim)` recomputed per element
- **Where:** `tensor.py:134-138` (verified), and same shape at `146-149`, `170-177`, `252-258`
  `(latter three unverified lines — confirm)`.
- **Verified (tensor):**
  ```python
  target = [
      i if i < len(self.dom) or i >= len(self.dom @ other.dom @ self.cod)
      else i - len(self.cod) if i >= len(self.dom @ self.cod)
      else i + len(other.dom) for i in source]
  ```
  Each iteration recomputes `Dim.__matmul__` (allocates a new `Dim`, concatenates tuples) for
  `self.dom @ other.dom @ self.cod` and `self.dom @ self.cod`.
- **Fix:** Hoist `n_self_dom = len(self.dom)`, `n_self_cod = len(self.cod)`, `n_other_dom =
  len(other.dom)`, and the two boundary sums **before** the comprehension.
- **Blockers:** None semantically — pure CSE. Verify the axis permutation is identical (doctests).
- **Grep:** `for i in source`, `len(self.dom @`, `np.moveaxis`.

### T2.5 — `Tensor.id` / `cup_factory` build a `Matrix` then unwrap `.array`
- **Where:** `tensor.py:114-116` (`id`), `~155-159` (`cup_factory`, *unverified lines*).
- **Verified (id):** `return cls(Matrix.id(product(dom.inside)).array, dom, dom)` — constructs a full
  `Matrix` object (with its own reshape) just to grab `.array` and rewrap as `Tensor`.
- **Fix:** Call the backend identity directly: `with backend() as np: arr = np.identity(n)`; or add a
  classmethod that builds the array without the intermediate Matrix object. Also `product(dom.inside)`
  is the recursive O(n²) `product` (T3.3) — `Dim` could expose a cached size.
- **Blockers:** dtype must match `cls.dtype`; `np.identity` defaults to float — pass `dtype=`.
- **Grep:** `Matrix.id(`, `def cup_factory`, `product(dom.inside)`.

### T2.6 — `Spider.spider_factory` writes the diagonal in a Python loop and pins numpy
- **Where:** `tensor.py:~189-191` `(unverified lines — confirm; verified region nearby)`.
- **Pattern (per sub-agent):**
  ```python
  result = cls.zero(dom, cod)
  for i in range(n):
      result.array[len(dom @ cod) * (i,)] = 1     # scalar assignment per i; pins backend('numpy')
  ```
  Per-element Python assignment; **in-place mutation breaks on immutable backends (jax)**; forces a
  numpy array even when the active backend is jax/torch (then reconverts).
- **Fix:** Build with a vectorized `np.eye(...)`/reshape **through the active backend**, or assemble
  on host then convert once. Remove the hard `backend('numpy')` pin.
- **Blockers:** Confirm exact lines and the `len(dom @ cod) * (i,)` index trick. The immutability
  issue means this may currently be **buggy under jax** — check `test/semantics` for a jax spider
  test (may be absent, which is itself a gap).
- **Grep:** `def spider_factory`, `result.array[`, `backend('numpy')`, `def zero`.

### T2.7 — `Matrix.copy` builds an O(x·n·x) Python nested-list comprehension
- **Where:** `matrix.py:325-329`.
- **Verified.**
  ```python
  array = [[i + int(j % n * x) == j for j in range(n * x)] for i in range(x)]
  return cls(array, x, n * x)
  ```
  ~`x·n·x` Python comparisons (e.g. x=64,n=8 ≈ 32k) producing a Python list, then `np.array`. Feeds
  `discard`/`merge`/`ones`.
- **Fix:** Construct with backend ops — this is a block/striped identity; `np.tile(np.eye(x), n)` or
  an `np.zeros` + fancy-index assignment, reshaped to `(x, n*x)`.
- **Blockers:** Must reproduce the exact boolean pattern (it's the Frobenius copy matrix) — verify
  the index identity `i + (j % n)*x == j` against a vectorized formulation with the doctest/`test`.
  Keep dtype = whatever `cls` is (often `bool`).
- **Grep:** `def copy`, `def discard`, `def merge`, `def ones`, `j % n`.

### T2.8 — `Matrix.repeat` / `Matrix.trace` use O(n²) Python matmuls + `sum` over Matrix objects
- **Where:** `matrix.py:359-371` (`repeat`), `373-389` (`trace`).
- **Verified.**
  ```python
  # repeat (boolean reflexive-transitive closure):
  return sum(self.id(self.dom).then(*n * [self]) for n in range(self.dom + 1))
  # trace: builds A,B,C,D via a generator of triple-matmuls, then A + (B >> D.repeat() >> C)
  ```
  `repeat` composes `n` copies for each `n ≤ dom` and sums **Python `Matrix` objects** (each `__add__`
  re-`__init__`s → reshape, T2.1). `trace` builds 6+ intermediate matrices.
- **Fix:** `repeat` is boolean closure → compute with repeated squaring / `np.linalg.matrix_power`
  accumulation, or a single boolean Floyd–Warshall on the ndarray. `trace` → `np.trace(self.array,
  axis1=..., axis2=...)` (or boolean equivalent), avoiding the row/column-vector matmul gymnastics.
- **Blockers:** **Restricted to `dtype == bool` and `dom == cod`** (guard at matrix.py:368) — this is
  the *relational* semiring, so closure semantics (logical OR, not arithmetic +) must be preserved;
  `sum(...)` here relies on bool `+` = OR. A `matrix_power` rewrite must use boolean matmul. Niche
  path (boolean matrices only) — lower priority but easy to get subtly wrong. Verify doctests
  matrix.py:365-366, 382.
- **Grep:** `def repeat`, `def trace`, `MATRIX_REPEAT_ERROR`, `D.repeat()`.

### T2.9 — `Matrix.map` iterates every element through Python
- **Where:** `matrix.py:315-317`.
- **Verified.** `array = list(map(func, self.array.reshape(-1)))`. Used by `subs`/`grad`/`lambdify`.
- **Fix:** For numeric dtypes, try `func(self.array)` directly (works when `func` is vectorized) and
  fall back to per-element only for object/sympy dtype.
- **Blockers:** `func` may be a scalar-only sympy callable → can't always vectorize. Gate on dtype or
  `try/except`. Verify `test/` symbolic (sympy) cases.
- **Grep:** `def map`, `self.array.reshape(-1)`, `def subs`, `def grad`, `def lambdify`.

### T2.10 — `Diagram.eval` builds a fresh `Functor` every call
- **Where:** `tensor.py:407-427`.
- **Verified.** `return Functor(ob=lambda x: x, ar=lambda f: f.array, dtype=dtype)(self)` — new
  `Functor` (and its `Category(...)` plumbing) per `eval`. Costly in training loops / parameter
  sweeps.
- **Fix:** Cache a module-level / per-class default evaluation Functor keyed on `dtype`.
- **Blockers:** `dtype` varies; cache per dtype. The `ar=lambda f: f.array` closure is stateless —
  safe to reuse. The `contractor` branch (tensor.py:426) is separate.
- **Grep:** `def eval`, `Functor(\n`, `ar=lambda f: f.array`.

### T2.11 — Other numerical smells
- `Matrix.__eq__` (`matrix.py:163-167`, verified) ends with `(self.array == other.array).all()` —
  for jax/torch this forces materialization; OK but note no `np.array_equal` fast-path and **no
  `__hash__`** (Matrix is unhashable → can't memoize by Matrix). **Grep:** `def __eq__` (matrix).
- `Matrix.__repr__` (`matrix.py:~216`, *unverified*) calls `.numpy()` on torch tensors (host
  transfer) and globally mutates `numpy.set_printoptions`. Avoid in logging. **Grep:** `def __repr__`
  (matrix), `set_printoptions`.
- `set_backend` (`matrix.py:513`, verified) mutates `backend.__wrapped__.__defaults__[1][-1]` —
  brittle + global (subsumed by T0.6's ContextVar fix). **Grep:** `__wrapped__`.
- `to_quimb`/`to_tn` call `box.eval()` (rebuilds a Functor, T2.10) instead of `box.array`
  (`tensor.py:~451-480, ~524`, *unverified*). **Grep:** `box.eval()`, `def to_quimb`, `def to_tn`.
- `Diagram.grad` (`tensor.py:~534-539`, *unverified*) recurses building two diagrams per box → O(n²)
  diagram allocs; `self.free_symbols` likely walks the whole diagram each call. **Grep:** `def grad`,
  `free_symbols`.

---

## TIER 3 — Core scaffolding & helpers (`cat.py`, `monoidal.py`, `utils.py`)

### T3.1 — `monoidal.Diagram.__init__` walks `inside` three times
- **Where:** `monoidal.py:518-522` (verified) → its `assert_isinstance(layer, Layer)` loop +
  `super().__init__` (cat) which loops twice more (T0.1).
- **Fix:** Pass `_scan=False` from trusted internal callers; do the Layer-type check only when
  scanning. Subsumed by the T0.1 `_scan` discipline.
- **Grep:** `class Diagram` (monoidal), `assert_isinstance(layer`.

### T3.2 — `Layer.__init__` folds dom/cod via repeated `@` and builds a name string every time
- **Where:** `monoidal.py:346-361`.
- **Verified.**
  ```python
  name, dom, cod = "", left[:0], left[:0]
  for i, box_or_typ in enumerate(self.boxes_or_types):
      ... dom, cod = dom @ x.dom, cod @ x.cod ...    # O(arity) Ty.__matmul__ allocs
      name += ... str(box_or_typ) ...                # builds a display string on the hot path
  ```
  Every `Diagram.tensor`/`then` produces fresh `Layer`s, each paying O(arity) `Ty` concatenations +
  `str()`. **Bug-adjacent:** L353 asserts `box` (the original middle param) not `box_or_typ`, so
  multi-box (`*more`) layers don't validate their extra boxes.
- **Fix:** Add a fast Layer constructor that accepts precomputed `dom`/`cod` and defers `name` (make
  `name` a lazy `@property`/`cached_property` built from `boxes_or_types` on demand — it's only used
  by `__repr__`/drawing).
- **Blockers:** `name` is passed to `Ob.__init__`; some code may read `layer.name`. Making it lazy
  requires auditing readers. The L353 assert bug should be fixed to `box_or_typ` regardless.
- **Grep:** `class Layer`, `boxes_or_types`, `def __matmul__` (Layer), `head @ other`.

### T3.3 — `utils.product` is recursive with `xs[1:]` → O(n²) + recursion overhead
- **Where:** `utils.py:205-214`.
- **Verified.** `return unit if not xs else product(xs[1:], unit * xs[0])` — new list slice per level.
  Called by `Tensor.__init__`/`id` (T2.1/T2.5) via `product(dom.inside)`.
- **Fix:** `functools.reduce(operator.mul, xs, unit)` (O(n), iterative). Better: cache `Dim`'s product
  on the `Dim` object so tensor code never recomputes it.
- **Blockers:** `unit` may be a list (doctest utils.py:212 `product([1,2,3], unit=[42])`), so use
  `reduce` with the same left-fold `unit * x` semantics. Trivial.
- **Grep:** `def product`, `product(`, `dom.inside`.

### T3.4 — `utils.inductive` recursive + `assert_isinstance` every step
- **Where:** `utils.py:401-411`.
- **Verified.** `return method(induction_step(self), n_steps - 1)` with `assert_isinstance(n_steps,
  int)` re-checked at each level. Used for `.l`/`.r` adjoint chains.
- **Fix:** Iterative `for _ in range(n_steps): self = induction_step(self); return self`; check
  `n_steps` once.
- **Blockers:** None. Watch the `n_steps < 0` ValueError and `n_steps == 0` identity branches.
- **Grep:** `def inductive`, `induction_step`, `\.l\b`, `\.r\b`.

### T3.5 — `symmetric.permutation` is O(n²·log n): rebuilds + re-validates every recursion
- **Where:** `symmetric.py:182-200`.
- **Verified.**
  ```python
  if list(range(len(dom))) != sorted(xs):              # O(n log n) — re-run every recursive call
      raise ValueError(messages.WRONG_PERMUTATION.format(len(dom), xs))
  ...
  >> dom[i] @ cls.permutation(
      [x - 1 if x > i else x for x in xs[1:]],          # O(n) new list per level
      dom[:i] + dom[i + 1:])                            # O(n) Ty concat per level
  ```
  Depth-n recursion → O(n² log n) just for revalidation + list/Ty rebuilds.
- **Fix:** Validate once at a public entry, then call an unchecked recursive/iterative inner. Operate
  on integer indices and slice once; or build the permutation diagram via a known O(n log n)
  decomposition.
- **Blockers:** The recursive structure encodes a specific braiding/swap decomposition — the output
  *diagram* must be identical (it's not just "a" permutation, it's a canonical one). Verify against
  symmetric doctests and `test/syntax`.
- **Grep:** `def permutation`, `sorted(xs)`, `WRONG_PERMUTATION`, `cls.swap`.

### T3.6 — `Layer.__eq__`/`__hash__` materialize `tuple(self)`; `Layer.boxes` re-slices per call
- **Where:** `monoidal.py:374-378` (`__eq__`/`__hash__`), `367-369` (`boxes`).
- **Verified.** `tuple(self)` triggers `__iter__` to rebuild a tuple that already exists as
  `self.boxes_or_types`. `boxes` returns `list(self.boxes_or_types[1::2])` fresh each call (summed by
  `Diagram.boxes`, T0.2).
- **Fix:** `__eq__`: `self.boxes_or_types == other.boxes_or_types`; `__hash__`:
  `hash(self.boxes_or_types)`; cache `boxes` as `cached_property`.
- **Blockers:** Keep the `isinstance(other, type(self))` guard. Consistent with T0.3 hash changes.
- **Grep:** `tuple(self)`, `boxes_or_types[1::2]`, `def boxes` (Layer).

### T3.7 — `Functor.__call__` linear right-fold rebuilds an arrow per box
- **Where:** `cat.py:909-913` (Arrow), `monoidal.py:1121-1126` (Layer) `(verified region per
  sub-agent — line-confirm)`.
- **Pattern:**
  ```python
  result = self.cod.ar.id(self(other.dom))
  for box in other.inside: result = result >> self(box)   # each >> → then → __init__ scan (T0.1)
  ```
  Applying a functor to N boxes → O(N²) tuple allocs + N validations.
- **Fix:** Accumulate `inside` into a list and build one arrow with `_scan=False` at the end.
- **Blockers:** Needs T0.1's trusted constructor; preserve functor semantics for `Sum`/`Bubble`
  branches. Verify `test/` functor cases.
- **Grep:** `def __call__` (Functor), `result = result >>`, `result @ self(`.

### T3.8 — `Ty.__pow__` goes through variadic `tensor` with per-arg `assert_isinstance`
- **Where:** `monoidal.py:198-199`; contrast the correct `PRO.__pow__` at `monoidal.py:290-291`.
- **Verified.** `return self.factory().tensor(*n_times * [self])` — allocates an n-list, an `*args`
  tuple, and runs `assert_isinstance` n times. `PRO` does `self.factory(n_times * self.n)` directly.
- **Fix:** `Ty.__pow__` → `self.factory(*(self.inside * n_times))` (one tuple, no per-element checks).
- **Blockers:** Must match existing semantics for `n_times == 0` (empty Ty). Trivial.
- **Grep:** `def __pow__`, `n_times \* \[self\]`.

### T3.9 — `Ty.__iter__` and `Arrow.__getitem__[int]` allocate a wrapper per index
- **Where:** `monoidal.py:189-191` (`Ty.__iter__` yields `self[i]`), `cat.py:258-263`
  (`__getitem__` int branch recurses to `self[key:key+1]`).
- **Verified.** Iterating a length-N `Ty` builds N single-object `Ty`s; `arrow[i]` builds a
  one-box Arrow via the slice path.
- **Fix:** `Ty.__iter__` → `yield from self.inside` (or yield `self.factory(x)` only if callers need
  a Ty — check usage; many want the `Ob`). For `Arrow.__getitem__[int]`, consider returning
  `self.inside[key]` where a Box is acceptable.
- **Blockers:** **Semantics change risk** — downstream code may rely on iterating a `Ty` yielding
  length-1 `Ty`s (not bare `Ob`s). Audit all `for x in some_ty` and `ty[i]` sites before changing;
  this one is easy to get wrong. The `Arrow.__getitem__` change similarly affects whether `d[i]` is a
  Box or a Diagram. **Lower priority / higher risk.**
- **Grep:** `def __iter__` (Ty), `def __getitem__` (cat Arrow), `self[key:key + 1]`.

### T3.10 — Misc constant-factor smells
- `cat.Bubble.__hash__` (`cat.py:725-727`, *unverified lines*) does 5 `getattr` + rebuilds a list
  literal per call → `hash((self.args, self.dom, self.cod, self.name, self.method))`. **Grep:**
  `getattr(self, x) for x`.
- `Box.__eq__` non-Box fallback (`cat.py:544-552`, verified) builds `self >> self.id(self.cod)` to
  compare against an Arrow — allocates on every box-vs-arrow comparison. **Grep:** `cast box as
  diagram`.
- `monoidal.Diagram.interchange` checks layer arity via `len(list(layer)) != 3` per layer
  (`monoidal.py:~779`, *unverified*) → use `len(layer.boxes_or_types)`; non-adjacent interchange
  recurses rebuilding via four `>>` per step (`~785-818`). **Grep:** `len(list(layer))`, `def
  interchange` (monoidal).
- `Layer.merge` → `Diagram.normal_form` (uses `str(diagram)` cache key, monoidal.py:878-885 verified)
  per layer pair in `foliation` → O(L²) full normalizations. **Grep:** `def merge`, `def foliation`,
  `def normal_form`, `str(diagram)`.
- `Sum.then`/`Sum.tensor` (`cat.py:645-650`, `monoidal.py:968-975`, *unverified*) build n·m terms,
  each fully re-validated (T0.1). **Grep:** `for f in self.terms for g in`.
- `utils.from_tree` (`utils.py:258-263`, verified) does `import discopy` + `getattr` chain per node →
  cache `dict[factory_str, type]`. **Grep:** `def from_tree`, `import discopy`.
- `utils.NamedGeneric.__class_getitem__` (`utils.py:148-197`, verified region) computes
  `tuple(getattr(cls, attr, None) for attr in attributes)` **before** the cache check. **Grep:**
  `__class_getitem__`, `_cache`.

---

## TIER 4 — Other modules (frobenius / rigid) — line numbers VERIFIED where noted

- **`rigid.snake_removal`** (`rigid.py:395-509`, **verified**): `find_snake` scans all boxes and
  calls `follow_wire`, which indexes the **quadratic** `diagram.boxes[i]`/`offsets[i]` properties
  (T0.2) **inside its `while` loop** (rigid.py:430,448) → each `follow_wire` is ~O(L³); `find_snake`
  adds an O(L) factor; the outer `while True` (rigid.py:501) **re-runs `find_snake` from scratch**
  after every removal. `unsnake` (462-498) does O(obstructions²) index bumping and slices the whole
  `inside` per snake (496). **Net: cubic-or-worse normalization.** Biggest pure-algorithm offender.
  - **Fix:** fix T0.2 first (kills the worst factor); hoist `boxes`/`offsets` in `follow_wire`;
    resume `find_snake` from the last position instead of restarting; batch `interchange` rewrites.
  - **Blockers:** snake removal must yield the **same rewrite sequence** (doctest rigid.py:409-412
    prints every step) — restructuring the search may change intermediate steps even if the final
    normal form matches. Tread carefully; keep the step-by-step doctest green.
  - **Grep:** `def snake_removal`, `def follow_wire`, `def find_snake`, `def unsnake`,
    `diagram.boxes[`, `while True`.
- **`rigid.Box.__hash__`** (rigid.py:587-588, **verified**) `hash(cat.Arrow.__repr__(self))` — see
  T0.3. **Grep:** `cat.Arrow.__repr__`.
- **`rigid.nesting`** (`~768-786`, unverified): recursive with `left[1:]`/`right[:-1]` Ty slicing per
  level → O(n²) cups/caps over an n-wire bundle. **Grep:** `def nesting`.
- **`rigid.Functor.__call__`** (`~746-765`, unverified): 5 `isinstance` checks/call; `transpose`/`ev`
  allocate fresh cup+cap diagrams per rotated box (`Box.rotate` called `|z|` times). **Grep:** `def
  transpose`, `def rotate`, `\.l\b`, `\.r\b`.
- **`frobenius.interleaving`** (`~333-357`, unverified): rebuilds whole diagrams per swap with
  `result <<= ... swap ...`; Ty-slicing per step → O(|typ|·(in+out)·|dom|). **Grep:** `def
  interleaving`, `result <<=`, `result >>=`.
- **`frobenius.coherence`** (`~383-400`, unverified): recursion with no memoization (two unshared
  descents); `x[::-1]`/`rotate()` on sub-diagrams. **Grep:** `def coherence`, `x[::-1]`.
- **`frobenius`/`Functor` spider construction** rebuilds spiders via `interleaving` each functor
  application — memoize by `(in, out, typ)`. **Grep:** `\.spiders(`, `def unfuse`.

---

## Correctness issues found en route (NOT performance — flag separately)

- **`monoidal.Layer.free_symbols`** (`monoidal.py:392-394`, **verified**):
  ```python
  return {x for _, box, _ in self.inside for x in box.free_symbols}
  ```
  `Box.inside == (self,)` (cat.py:490), so this iterates `(self,)` and unpacks the **Layer** into
  `(_, box, _)`. **Works** for standard 3-element layers (returns the single box's symbols) but
  **raises `ValueError: too many values to unpack`** for multi-box (foliated) layers with `*more`.
  Either handle arity (`for box in self.boxes_or_types[1::2]`) or document the 3-element assumption.
- **`monoidal.Layer.__init__:353`** (verified): `assert_isinstance(box, Box)` checks the original
  `box` parameter, not the loop's `box_or_typ` — so extra boxes in a multi-box layer are never
  type-checked.
- **`hypergraph.from_callable`** missing `try/finally` around the `__call__` monkeypatch (T1.8) — a
  raising `func` leaves the arrow class clobbered.

---

## Suggested fix order (Pareto)

1. **T0.1 + T0.7** — soundness-preserving `then` rewrite (lift join-check into `then`, then
   `_scan=False`) + drop `Id(dom)`/`Id(cod)` (every composition).
2. **T0.2** — cache/hoist `Diagram.boxes`/`offsets` (turns `normalize` & `snake_removal` from
   ~cubic to linear; biggest single algorithmic win).
3. **T0.4 + T0.5** — memoize `factory_name`, defer `assert_isinstance` formatting (library-wide).
4. **T0.3** — replace `hash(repr(self))` everywhere (every dict/set op) — mind the equality contract.
5. **T1.1 + T1.2** — hypergraph `__init__` quadratics + cache `spider_wires`/`ports` (every
   hypergraph op).
6. **T0.6 + T2.1 + T2.2** — backend singleton/ContextVar; trusted Matrix/Tensor constructor; cache
   `Box.array` (every numerical op).
7. **Algorithms:** T1.3/T1.6 (simplify/make_*), T3.5 (permutation), Tier-4 `snake_removal`.
8. **Vectorize:** T2.6/T2.7/T2.8/T2.9 (spider_factory/copy/repeat/trace/map), T2.3/T2.4 (Functor &
   Tensor loop invariants).
9. **Cleanups:** T3.2/T3.6/T3.8 (Layer & Ty), T2.5/T2.10 (Tensor.id/eval), T3.10 misc.

**Before any fix:** add a benchmark (none exist) and run the doctests + `test/` suite — they are the
only correctness guardrail.
