cubic-dev-ai review on PR #644, two findings on `discopy/hypergraph.py`:

**Finding 1** (`Hypergraph.from_glued`, ~line 1424):
> When successive images change the scan width, this slice assignment still
> moves every later scan entry. Repeated insertions at the front, such as
> state layers, remain quadratic; batch updates or use an index structure
> that avoids repeated middle-list shifts.

**Finding 2** (`Hypergraph.from_diagram`, ~line 1466):
> When converting a diagram whose category differs from the requested
> Hypergraph target, this selects the target functor instead of the source
> factory's functor. Use `factory.functor` unconditionally so source-specific
> structural box mappings are preserved.

- [x] Investigated finding 1 with timing measurements (isolated the scan
      union-find loop from the rest of the pipeline). The front-insertion
      pattern is genuinely O(n^2) in isolation (measured ~4x per doubling,
      91s at n=800,000), but in the real `from_diagram`/`from_glued`
      pipeline, `Hypergraph.__init__`'s own per-box validation/`ports`
      overhead dominates by 1-2 orders of magnitude up to n in the
      hundreds of thousands (measured: prepend and append patterns track
      each other within ~15% up to n=256,000, both already needing over a
      minute). Not fixed: no evidence this is a practical bottleneck at
      any diagram size this library is used at, and an index structure
      would be over-engineering for a cost this deeply dominated
      elsewhere. Explained on the PR thread.
- [x] Investigated finding 2 with a concrete reproduction: a
      `symmetric.Hypergraph.from_diagram` call on a `frobenius.Diagram`
      containing a `Spider` box used `cls.functor` (`symmetric.Functor`,
      which does not know how to expand a `Spider`) instead of
      `factory.functor` (`frobenius.Functor`, which does), flattening the
      spider into an opaque box instead of pure wiring. Confirmed real
      and fixed: `functor = factory.functor(...)` unconditionally, matching
      cubic's suggestion and the pre-existing `_naive_from_diagram` test
      oracle, which already only ever used `factory.functor`. Added
      `test_Hypergraph_from_diagram_pinned_category` to `test/hypergraph.py`,
      verified it fails without the fix and passes with it.
