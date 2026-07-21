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
