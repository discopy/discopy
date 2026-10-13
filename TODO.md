# TODO

USER's review of 2026-10-13, verbatim:

> Change of terminology: let's rename this optics module into lens so that we don't confuse it with quantum optics / photonics as the optyx lbrary will move here @giodefelice
>
> "Monoidal lens" is used more or less interchangeably with "monoidal optics" so let's use that terminology and reproduce the same architecture as in the `para` module, i.e. one class for each doctrine where is makes sense e.g. `lens.Symmetric`, `lens.Traced`, `lens.Markov` etc. (so e.g. the `Prism` class we filed as an issue will be `lens.Comarkov`)

> can't this be just a dunder for unary minus of a Ty?

(on `negatives = staticmethod(reversed)` in `discopy/interaction.py`)

- [x] rename `discopy.optics` to `discopy.lens`: `Optic` → `lens.Symmetric`, `Traced` → `lens.Traced`, `Lens` → `lens.Markov`, `to_lens`/`to_optic` → `to_markov`/`to_symmetric`; tests, images, docs, changelog follow
- [x] answer the `negatives` thread: no dunder fits without going in a circle, asked USER for **A** (`lens.Ty` overrides `tensor`, `interaction.py` back to `main`) or **B** (keep a hook, renamed)
- [ ] apply USER's answer on the `negatives` thread
- [ ] apply USER's A/B on `lens.Markov` as a subclass of `lens.Symmetric`, asked on the PR
- [x] retitle #757 (`Prism`) as `lens.Comarkov`
