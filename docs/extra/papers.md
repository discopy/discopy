# Papers

A list of academic papers that make use of DisCoPy.

## DisCoPy: Monoidal categories in Python

https://doi.org/10.4204/EPTCS.333.13

Giovanni de Felice, Alexis Toumi, Bob Coecke

**Abstract:**
We introduce DisCoPy, an open source toolbox for computing with monoidal categories. The library provides an intuitive syntax for defining string diagrams and monoidal functors. Its modularity allows the efficient implementation of computational experiments in the various applications of category theory where diagrams have become a lingua franca. As an example, we used DisCoPy to perform natural language processing on quantum hardware for the first time.

## Diagrammatic Differentiation for Quantum Machine Learning

https://doi.org/10.4204/EPTCS.343.7

Alexis Toumi, Richie Yeung, Giovanni de Felice

**Abstract:**
We introduce diagrammatic differentiation for tensor calculus by generalising the dual number construction from rigs to monoidal categories. Applying this to ZX diagrams, we show how to calculate diagrammatically the gradient of a linear map with respect to a phase parameter. For diagrams of parametrised quantum circuits, we get the well-known parameter-shift rule at the basis of many variational quantum algorithms. We then extend our method to the automatic differentation of hybrid classical-quantum circuits, using diagrams with bubbles to encode arbitrary non-linear operators. Moreover, diagrammatic differentiation comes with an open-source implementation in DisCoPy, the Python library for monoidal categories. Diagrammatic gradients of classical-quantum circuits can then be simplified using the PyZX library and executed on quantum hardware via the tket compiler. This opens the door to many practical applications harnessing both the structure of string diagrams and the computational power of quantum machine learning.

## DisCoPy for the quantum computer scientist

https://arxiv.org/abs/2205.05190

Alexis Toumi, Giovanni de Felice, Richie Yeung

**Abstract:**
DisCoPy (Distributional Compositional Python) is an open source toolbox for computing with string diagrams and functors. In particular, the diagram data structure allows to encode various kinds of quantum processes, with functors for classical simulation and optimisation, as well as compilation and evaluation on quantum hardware. This includes the ZX calculus and its many variants, the parameterised circuits used in quantum machine learning, but also linear optical quantum computing. We review the recent developments of the library in this direction, making DisCoPy a toolbox for the quantum computer scientist.

## Functorial Language Models

https://arxiv.org/abs/2103.14411

Alexis Toumi, Alex Koziell-Pipe

**Abstract:** We introduce functorial language models: a principled way to compute probability distributions over word sequences given a monoidal functor from grammar to meaning. This yields a method for training categorical compositional distributional (DisCoCat) models on raw text data. We provide a proof-of-concept implementation in DisCoPy, the Python toolbox for monoidal categories.
