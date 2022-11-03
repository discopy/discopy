
![snake equation](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/snake-equation.png)

# DisCoPy

[![build](https://github.com/oxford-quantum-group/discopy/actions/workflows/build_test.yml/badge.svg)](https://github.com/oxford-quantum-group/discopy/actions/workflows/build_test.yml)
[![readthedocs](https://readthedocs.org/projects/discopy/badge/?version=main)](https://discopy.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/discopy.svg)](https://badge.fury.io/py/discopy)
[![DOI: 10.4204/EPTCS.333.13](http://img.shields.io/badge/DOI-10.4204/EPTCS.333.13-brightgreen.svg)](https://doi.org/10.4204/EPTCS.333.13)

DisCoPy is a Python toolkit for computing with string diagrams.

* **Documentation:** https://discopy.readthedocs.org/
* **Source code:** https://github.com/oxford-quantum-group/discopy
* **Paper (for category theorists):** https://doi.org/10.4204/EPTCS.333.13
* **Paper (for quantum computer scientists):** https://arxiv.org/abs/2205.05190

It was first developed to implement [quantum natural language processing](https://arxiv.org/abs/2012.03755).
This use case is now packaged in its own library, [lambeq](https://github.com/CQCL/lambeq/).
While most of the current features focus on quantum computing
([circuits](https://discopy.readthedocs.io/en/main/discopy/quantum.circuit.html),
[ZX calculus](https://discopy.readthedocs.io/en/main/discopy/quantum.zx.html),
[photonics](https://discopy.readthedocs.io/en/main/discopy/quantum.optics.html)) and natural language processing ([pregroup](https://discopy.readthedocs.io/en/main/discopy/grammar.pregroup.html), [CCG](https://discopy.readthedocs.io/en/main/discopy/grammar.ccg.html), [formal grammars](https://discopy.readthedocs.io/en/main/discopy/grammar.html)),
the toolkit is flexible enough for all the applications of string diagrams:
[neural networks](https://arxiv.org/abs/1711.10455),
[probabilistic programs](https://arxiv.org/abs/1908.07021),
[concurrent processes](https://hal.archives-ouvertes.fr/hal-02134182/),
[electric circuits](https://arxiv.org/abs/2106.07763),
[database queries](https://arxiv.org/abs/1804.07626),
[logical formulae](https://link.springer.com/chapter/10.1007/978-3-030-54249-8_32), etc.

It provides:

* a flexible data structure for diagrams in (pre)monoidal categories
* a hierarchy of classes for graphical gadgets (swaps, cups and caps, etc.)
* methods for composing, drawing and rewriting diagrams
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
