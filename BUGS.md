# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.quantum.zx.Diagram`, stacked on
`split/4-tensor` since `zx.Diagram` subclasses `tensor.Diagram`. See the
earlier branches' BUGS.md for the rest of the property suite's history.

## Serialisation inherited with the wrong signature

- `zx.Spider`, `Scalar` and `H` inherited `to_tree`/`from_tree` and
  `repr` from `tensor.Box`/`Bubble`, whose `(name, dom, cod)` keys their
  own `__init__` rejects — so `eval(repr(x))`, `dumps`/`loads` or both
  crashed. Fixed by giving each its own serialisation reading the data it
  actually needs (phase, scalar value); `Scalar`'s was found by a late
  rare draw after the rest were already fixed.

## Pickling that loses or demands state

- `zx.H` carried a `lambda` as its own dagger, unpicklable by
  construction. It is now a `Hadamard` class that is its own dagger.

## Open, declared and recorded in the matrix

- `zx.Diagram` inherits `tensor`'s `spider_factory`, which expects
  dimensions rather than `PRO` types, so a functor cannot rebuild a ZX
  spider — xfailed in `proptest/test_normal_form.py`.
