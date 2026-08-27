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

- [ ] Investigate finding 1 with a concrete timing reproduction; fix if it is
      a real practical bottleneck, otherwise explain why not with evidence.
- [ ] Investigate finding 2 with a concrete reproduction; fix if real.
