
![snake equation](https://github.com/discopy/discopy/raw/main/docs/_static/snake-equation.png)

# DisCoPy

[![build](https://github.com/discopy/discopy/actions/workflows/build_test.yml/badge.svg)](https://github.com/discopy/discopy/actions/workflows/build_test.yml)
[![readthedocs](https://readthedocs.org/projects/discopy/badge/?version=main)](https://docs.discopy.org/)
[![PyPI version](https://badge.fury.io/py/discopy.svg)](https://badge.fury.io/py/discopy)
[![DOI: 10.4204/EPTCS.333.13](http://img.shields.io/badge/DOI-10.4204/EPTCS.333.13-brightgreen.svg)](https://doi.org/10.4204/EPTCS.333.13)

DisCoPy is a Python toolkit for computing with [string diagrams](https://en.wikipedia.org/wiki/String_diagram).

* **Organisation:** https://discopy.org
* **Documentation:** https://docs.discopy.org
* **Source code:** https://github.com/discopy/discopy
* **Paper (for applied category theorists):** https://doi.org/10.4204/EPTCS.333.13
* **Paper (for quantum computer scientists):** https://arxiv.org/abs/2205.05190

DisCoPy began as an implementation of [DisCoCat](https://en.wikipedia.org/wiki/DisCoCat) and [QNLP](https://en.wikipedia.org/wiki/Quantum_natural_language_processing). This has now become its own library: [lambeq](https://cqcl.github.io/lambeq).

## Features

* an `Arrow` data structure for free [dagger categories](https://en.wikipedia.org/wiki/Dagger_category) with formal sums, unary operators and symbolic variables from [SymPy](https://www.sympy.org/en/index.html)
* a `Diagram` data structure for planar string diagrams in any ([pre](https://ncatlab.org/nlab/show/premonoidal+category))[monoidal category](https://en.wikipedia.org/wiki/Monoidal_category) in the [hierarchy of graphical languages](https://en.wikipedia.org/wiki/String_diagram#Hierarchy_of_graphical_languages) (with braids, twists, spiders, etc.)
* a `Hypergraph` data structure for string diagrams in hypergraph categories and its restrictions to symmetric, traced, compact and Markov categories
* methods for diagram composition, drawing, rewriting and `Functor` evaluation into:
  - Python code, i.e. wires as types and boxes as functions
  - [tensor networks](https://en.wikipedia.org/wiki/Tensor_network), i.e. wires as dimensions and boxes as arrays from [NumPy](https://numpy.org), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [TensorNetwork](https://github.com/google/TensorNetwork) and [JAX](https://github.com/google/jax)
* an implementation of [categorical quantum mechanics](https://en.wikipedia.org/wiki/Categorical_quantum_mechanics) interfacing with:
  - [tket](https://github.com/CQCL/tket) for circuit compilation
  - [PyZX](https://github.com/Quantomatic/pyzx) for optimisation with the [ZX calculus](https://zxcalculus.com/)
  - [PennyLane](https://pennylane.ai/) for automatic differentiation
* an implementation of formal grammars ([context-free](https://en.wikipedia.org/wiki/Context-free_grammar), [categorial](https://en.wikipedia.org/wiki/Categorial_grammar), [pregroup](https://en.wikipedia.org/wiki/Pregroup_grammar) or [dependency](https://en.wikipedia.org/wiki/Dependency_grammar)) with interfaces to [lambeq](https://cqcl.github.io/lambeq), [spaCy](https://spacy.io/) and [NLTK](https://www.nltk.org/)

## Architecture

Software dependencies between modules go top-to-bottom, left-to-right and [forgetful functors](https://en.wikipedia.org/wiki/Forgetful_functor) between categories go the other way.

[![architecture](https://github.com/discopy/discopy/raw/main/docs/api/architecture.png)](https://docs.discopy.org#architecture)

## Quickstart

```shell
pip install discopy
```

If you want to see DisCoPy in action, check out the [QNLP tutorial](https://docs.discopy.org/en/main/notebooks/qnlp.html)!

## Contribute

We're keen to welcome new contributors!

First, read the [contributing guidelines](https://github.com/discopy/discopy/blob/main/CONTRIBUTING.md).
Then get in touch on [Discord](https://discopy.org/discord)
or [open an issue](https://github.com/discopy/discopy/issues/new).

## How to cite

If you wish to cite DisCoPy in an academic publication, we suggest you cite:

* G. de Felice, A. Toumi & B. Coecke, _DisCoPy: Monoidal Categories in Python_, EPTCS 333, 2021, pp. 183-197, [DOI: 10.4204/EPTCS.333.13](https://doi.org/10.4204/EPTCS.333.13)

If furthermore your work is related to quantum computing, you can also cite:

* A. Toumi, G. de Felice & R. Yeung, _DisCoPy for the quantum computer scientist_, [arXiv:2205.05190](https://arxiv.org/abs/2205.05190)
