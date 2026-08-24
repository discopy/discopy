implement the to_compact proposal suggested by toumix on the PR thread. make a closed.Diagram.to_compact method implemented as a functor which turns a closed diagram with curry bubbles to a diagram with coeval morphisms.

- [x] Implement and verify `closed.Diagram.to_compact` as proposed.

implement to_compact on cmap as well and recover the curry diagram shapes in the doctest as they were on the main branch

- [x] Implement `CMap.to_compact` and restore the curry doctest shapes.
