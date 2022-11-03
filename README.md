
![snake equation](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/snake-equation.png)

# DisCoPy

[![build](https://github.com/oxford-quantum-group/discopy/actions/workflows/build_test.yml/badge.svg)](https://github.com/oxford-quantum-group/discopy/actions/workflows/build_test.yml)
[![readthedocs](https://readthedocs.org/projects/discopy/badge/?version=main)](https://discopy.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/discopy.svg)](https://badge.fury.io/py/discopy)
[![arXiv:2005.02975](http://img.shields.io/badge/DOI-10.4204/EPTCS.333.13-brightgreen.svg)](
https://doi.org/10.4204/EPTCS.333.13)

DisCoPy is a Python toolkit for computing with string diagrams.

* **Documentation:** https://discopy.readthedocs.org/
* **Source code:** https://github.com/oxford-quantum-group/discopy
* **Paper (for category theorists):** https://doi.org/10.4204/EPTCS.333.13
* **Paper (for quantum computer scientists):** https://arxiv.org/abs/2205.05190

It provides:

* a flexible data structure for diagrams in (pre)monoidal categories
* methods for composing, drawing and rewriting diagrams
* encodings of [pregroup](https://en.wikipedia.org/wiki/Pregroup_grammar), [CCG](https://en.wikipedia.org/wiki/Combinatory_categorial_grammar) and [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar) as diagrams
* algorithms for functor application, i.e. interpreting diagrams as
  - tensor networks with [NumPy](https://numpy.org), [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/)
  - quantum circuits with [tket](https://github.com/CQCL/tket), [PyZX](https://github.com/Quantomatic/pyzx) and [PennyLane](https://pennylane.ai/)

## Install

```shell
pip install discopy
```

## Test

If you want the bleeding edge, you can install DisCoPy locally:

```shell
git clone https://github.com/oxford-quantum-group/discopy.git
cd discopy
pip install .
```

You should check you haven't broken anything by running the test suite:

```shell
pip install ".[test]" .
pip install pytest coverage pycodestyle
coverage run -m pytest --doctest-modules --pycodestyle
coverage report -m discopy/*.py discopy/*/*.py
```

The documentation is built using
[sphinx](https://www.sphinx-doc.org/en/master/).
You can build it locally with:

```shell
(cd docs && (make clean; make html))
```

## Contribute

Contributions are welcome, please get in touch or
[open an issue](https://github.com/oxford-quantum-group/discopy/issues/new).

## How to cite

If you wish to cite DisCoPy in an academic publication, we suggest you cite:

* G. de Felice, A. Toumi & B. Coecke, _DisCoPy: Monoidal Categories in Python_, EPTCS 333, 2021, pp. 183-197, [DOI: 10.4204/EPTCS.333.13](https://doi.org/10.4204/EPTCS.333.13)

If furthermore your work is related to quantum computing, you can also cite:

* A. Toumi, G. de Felice & R. Yeung, _DisCoPy for the quantum computer scientist_, [arXiv:2205.05190](https://arxiv.org/abs/2205.05190)
