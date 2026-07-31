# TODO

Task, as specified by @giodefelice (quoted verbatim):

> Check out this PR now https://github.com/discopy/discopy/blob/intertwiner-spaces/docs/notebooks/learning-intertwiners.ipynb. Can we find a better dataset for which we can learn intertwiners? It would be cool if we could learn/generate the fusion rules in TQFTs. There is a risk that we are already computing them in the parametrisation of the chart. I like the toric code/Kitaev ref but it should be deepened, in the quantum computing context, does learning intertwiners mean learning over error-correcting codes?

The circularity is real, twice over: the chart's parameter dimension is the aggregate
fusion multiplicity, computed before any data is seen, and the notebook's "unknown"
target is sampled from the chart's own slices, so the chart fit wins by construction.
The plan, agreed in session: replace the circular fit with a non-circular pipeline —
estimate the modular S-matrix from monodromy (braiding) data, get the fusion rules
from Verlinde's formula, cross-check against chart dimensions; deepen the Kitaev
reference into a covariant-QEC section; add the smallest non-abelian double D(k[S3])
so the generated fusion table is not a group law.

- [x] merge origin/main into the branch, adapt to the marimo notebook
      convention and the current hopf.py (the notebook conversion itself
      happens with the notebook rework below)
- [ ] `HopfAlgebra.symmetric(n)`: the group algebra of the symmetric group
- [ ] generalise `Representation.anyon` to the double of any group algebra:
      the irrep of a conjugacy class and an irrep of its centraliser
- [ ] tests: the 8 anyons of D(k[S3]) are valid modules with dims summing
      in squares to 36, S unitary, Verlinde integral, agreement with chart dims
- [ ] notebook: fusion rules from braiding, exactly — monodromy traces, S,
      Verlinde, cross-checked against per-channel chart dimensions
- [ ] notebook: learning the fusion rules from finite-shot simulated
      interferometry, rounding Verlinde to integers
- [ ] notebook: the D(k[S3]) fusion table, generated not assumed
- [ ] notebook: anyons as an error-correcting code — Hom(1, V1 (x) ... (x) Vn)
      as the code space via the chart, covariant encoder = intertwiner, references
- [ ] update the PR body
