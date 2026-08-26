# TODO

Human prompt, verbatim:

> open a follow up PR with this probabilistic token passing on lambda terms, make sure you use an
> example from the relevant literature and cite it explicitly (probably Dal Lago geometry of
> bayesian programming? open to other suggestions) https://github.com/discopy/discopy/issues/618

- [ ] describe the mathematics before writing code: what the token is, what its positions are,
      what one step is and why the execution formula is the additive trace
- [ ] `discopy.kleisli.token`: the probabilistic token machine for closed lambda terms, i.e. a
      `Channel[Subdistribution]` whose trace is the term's unnormalised semantics
- [ ] `sample` and `score` as the two effectful constants, following Dal Lago & Hoshino
- [ ] a worked example from the literature, cited explicitly
- [ ] fix the wrong arXiv number for the Dal Lago & Hoshino citation in `kleisli.additive`
- [ ] tests and 100% coverage of the new module
- [ ] `CHANGELOG.md` entry, `pflake8 discopy` and the full test suite green
