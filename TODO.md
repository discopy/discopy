# TODO

> fix the bugs, or find open issues for them like  zx.Z(1,1,0).to_hypergraph()
>
> zx.Equation unusable. metatheory noted it unreported on 07-05.
>
> Tr_U(f⊗id_U) = f False under symmetric.Equation

- [x] `rigid.PRO.unwind` returns itself, so `Hypergraph.__init__` accepts `PRO` spider types instead of raising `AttributeError: 'int' object has no attribute 'unwind'`
- [x] `frobenius.Functor.__call__` only turns a spider into `cod.spiders` when it is the structural spider of its category, so phased spiders and ZX `Z`/`X` boxes keep their data through `to_hypergraph` (and through any functor) instead of collapsing
- [x] Doctest + test for `zx.Diagram.to_hypergraph` and `zx.Equation`
- [x] `Tr_U(f ⊗ id_U) = f` is not a traced axiom (`Tr_U(id_U)` is a scalar); document the scalar-loop semantics of `to_hypergraph` in `traced`/`symmetric` and leave the behaviour
- [x] `CHANGELOG.md` entry, `pflake8`, `pytest`
