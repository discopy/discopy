# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.quantum.circuit.Circuit`, stacked on
`split/4-tensor` since `Circuit` subclasses `tensor.Diagram`. See the
earlier branches' BUGS.md for the rest of the property suite's history.

## Serialisation inherited with the wrong signature

- `quantum.gates.Ket`, `Bra` inherited `QuantumGate`'s repr, which took a
  bitstring but printed a name — fixed by giving each its own repr that
  round-trips the bitstring.

## Pickling that loses or demands state

- `quantum.circuit.Box.__setstate__` demanded a `_mixed` key that
  plumbing like `quantum.Swap` never stores, crashing on any pickled
  circuit built from such boxes. Fixed by only reading and renaming the
  key when it is actually present.

## Equality sensitive to representation noise

- `QuantumGate` equality compares reprs, and `complex(v)` keeps IEEE
  signed zeros, so numerically equal gates (`-1j` vs `(-0-1j)`) compared
  unequal. Fixed by normalising the zeros on construction
  (`complex(v) + 0j`).
