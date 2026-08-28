# TODO

Correctness review round on #677 (cubic, 9 findings), summarised:

> `discopy/closed.py` and `discopy/biclosed.py` can reuse names already
> present in free or annotated variables, causing variable capture or failed
> readback identity checks; generate fresh names that exclude all existing
> names.
> `discopy/cmap.py` can produce terms that silently drop free variables for
> rooted maps with disconnected boundary components, while
> `discopy/neural.py` can fail valid scalar networks via `torch.cat([])`;
> reject disconnected maps and provide an empty `(batch_size, 0)` port
> tensor.
> `docs/notebooks/neural-boolean.ipynb` does not reliably validate results:
> misses only print, `mux` is skipped, and `extract` truncates after 100
> nodes; assert failures, include an arity-3 probe, and remove or guard the
> traversal cap. The documented notebook workflow is incomplete:
> `neural-boolean.ipynb` requires an undeclared JAX dependency and
> `docs/notebooks/neural-church.md` links to an unpublished `.ipynb`.

- [x] Fresh names avoid
  capture: `closed.Substitution` and the `biclosed` readback skip names
  already present, with tests (closed.py:511, biclosed.py:105)
- [x] A portless scalar
  `Network` forwards an empty `(batch_size, 0)` tensor instead of crashing
  on `torch.cat([])` (neural.py:400)
- [x] `CMap.to_term`
  rejects a rooted map with a disconnected boundary component instead of
  dropping its free variables (cmap.py:1676)
- [x] The church
  notebook links the Boolean notebook's repository source, which the docs
  exporter does not publish (neural-church.md:8)
- [x] The four
  boolean-notebook findings (assert not print, `mux` eta probe, the
  `max_nodes` cap, the JAX dependency) fold into the queued marimo
  translation, filed as an issue and answered on the threads
