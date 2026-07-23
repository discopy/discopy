# TODO

Prompt ([#437](https://github.com/discopy/discopy/issues/437), verbatim):

> Instead of lists of odd length of length 3 minimum alternating with type, box, type, etc. we want the following way of defining layers: a  monoidal layer holds a list of boxes and non-empty types with at least one box and no two consecutive types.
>
> Whiskering a layer with a type on the left (right) only appends to the list if the type is non-empty and the left-most (right-most) element of the layer's internal list is a box.
>
> Tensoring a layer that ends with a type with a layer that begins with a type should tensor them so that the resulting layer has the length given by the sum of the lengths of the two layers minus 1.
>
> Initialising a new layer should scan=True by default to go through the list and tensor consecutive types to enforce the invariant, but when the layer is constructed by one of the methods above we use scan=False because we know we are preserving the invariant, so tensor a list of n layers takes linear time rather than quadratic.
>
> The same logic applies to #362 by replacing "type" with "permutation" in the discussion above: a symmetric layer holds a list of boxes and non-empty permutations with no two consecutive permutations with the following condition either a) it has at least one box or b) it is a singleton list of a non-identity permutation.

---

- [ ] Redefine `monoidal.Layer` on the new representation: boxes and non-empty types, at least one box, no two consecutive types
- [ ] Whiskering appends the type only when it is non-empty and the outermost element is a box, otherwise merges it into the boundary type
- [ ] Tensoring layers merges a trailing type with a leading type (resulting length = sum of lengths − 1)
- [ ] Constructor defaults to `scan=True` (rescans the list, merging consecutive types to restore the invariant); every internal call site that already preserves the invariant constructs with `scan=False`, so tensoring `n` layers is linear rather than quadratic
- [ ] `symmetric.Layer` (#362): the same invariant with "permutation" in place of "type", except it may also be a singleton list holding one non-identity permutation and no boxes
- [ ] Adjust `dom`/`cod`/`name` computation and `boxes_and_offsets` to the new representation
- [ ] Sweep dependent code (drawing, foliation, `symmetric.Layer` from #362) and update doctests + README
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`

## Guidance (🐦 birdsong, 2026-07-22)

- wait for #362 (symmetric-layer refactor, branch `claude/discopy-main-work-xu4vkj`) to land
  first. its own TODO defers this exact representation change to here, and
  `symmetric.Layer` subclasses `monoidal.Layer` — start now, rebase twice.
- #362 just fixed a `Layer.merge` crash on permutation layers. keep it fixed, add a
  regression test so this change can't reintroduce it.
- `boxes_and_offsets` feeds the drawing backend directly. no behaviour change for
  diagrams that already round-trip, or every drawing test breaks.
- eval(repr(x)) == x still has to hold on the new representation.

## Guidance (🌤️ daylight, 2026-07-22)

- #362 landed its refactor as of today, all points [x], undrafted, just waiting
  on your merge — so "wait for #362" now means wait for the merge, not the work.
  one rebase should do it.
- #444 (Swap ⊂ Permutation) got deferred out of #362, separate issue, doesn't
  block you.

## Guidance (🐦 birdsong, 2026-07-23)

- Alexis edited issue #437 (2026-07-22 16:00Z) to add the `scan=True`/`scan=False`
  linear-time requirement and the precise `symmetric.Layer` invariant (case a/b) —
  both added as their own checklist points above, and the verbatim prompt refreshed
  to match. Per `RULES.md` rule 1 this refresh is authorized by his direct
  instruction in today's session ("make sure the todos reflect any changes to the
  issues").
