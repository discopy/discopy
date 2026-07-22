# TODO

Prompt ([#437](https://github.com/discopy/discopy/issues/437), verbatim):

> Instead of lists of odd length of length 3 minimum alternating with type, box, type, etc. we want the following way of defining layers: a layer holds a list of boxes and non-empty types with at least one box and no two consecutive types.
>
> Whiskering a layer with a type on the left (right) only appends to the list if the type is non-empty and the left-most (right-most) element of the layer's internal list is a box.
>
> Tensoring a layer that ends with a type with a layer that begins with a type should tensor them so that the resulting layer has the length given by the sum of the lengths of the two layers minus 1.

---

- [ ] Redefine `monoidal.Layer` on the new representation: boxes and non-empty types, at least one box, no two consecutive types
- [ ] Whiskering appends the type only when it is non-empty and the outermost element is a box, otherwise merges it into the boundary type
- [ ] Tensoring layers merges a trailing type with a leading type (resulting length = sum of lengths − 1)
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
