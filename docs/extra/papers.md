# Publications

## PhD theses

### Category Theory for Quantum Natural Language Processing

https://arxiv.org/abs/2212.06615

Alexis Toumi

**Abstract:**
This thesis introduces quantum natural language processing (QNLP) models based on a simple yet powerful analogy between computational linguistics and quantum mechanics: grammar as entanglement. The grammatical structure of text and sentences connects the meaning of words in the same way that entanglement structure connects the states of quantum systems. Category theory allows to make this language-to-qubit analogy formal: it is a monoidal functor from grammar to vector spaces. We turn this abstract analogy into a concrete algorithm that translates the grammatical structure onto the architecture of parameterised quantum circuits. We then use a hybrid classical-quantum algorithm to train the model so that evaluating the circuits computes the meaning of sentences in data-driven tasks.

The implementation of QNLP models motivated the development of DisCoPy (Distributional Compositional Python), the toolkit for applied category theory of which the first chapter gives a comprehensive overview. String diagrams are the core data structure of DisCoPy, they allow to reason about computation at a high level of abstraction. We show how they can encode both grammatical structures and quantum circuits, but also logical formulae, neural networks or arbitrary Python code. Monoidal functors allow to translate these abstract diagrams into concrete computation, interfacing with optimised task-specific libraries.

The second chapter uses DisCopy to implement QNLP models as parameterised functors from grammar to quantum circuits. It gives a first proof-of-concept for the more general concept of functorial learning: generalising machine learning from functions to functors by learning from diagram-like data. In order to learn optimal functor parameters via gradient descent, we introduce the notion of diagrammatic differentiation: a graphical calculus for computing the gradients of parameterised diagrams.

### Categorical Tools for Natural Language Processing

https://arxiv.org/abs/2212.06636

Giovanni de Felice

**Abstract:**
This thesis develops the translation between category theory and computational linguistics as a foundation for natural language processing. The three chapters deal with syntax, semantics and pragmatics. First, string diagrams provide a unified model of syntactic structures in formal grammars. Second, functors compute semantics by turning diagrams into logical, tensor, neural or quantum computation. Third, the resulting functorial models can be composed to form games where equilibria are the solutions of language processing tasks. This framework is implemented as part of DisCoPy, the Python library for computing with string diagrams. We describe the correspondence between categorical, linguistic and computational structures, and demonstrate their applications in compositional natural language processing.

## Conferences

### DisCoPy: Monoidal categories in Python

https://doi.org/10.4204/EPTCS.333.13

Giovanni de Felice, Alexis Toumi, Bob Coecke

**Abstract:**
We introduce DisCoPy, an open source toolbox for computing with monoidal categories. The library provides an intuitive syntax for defining string diagrams and monoidal functors. Its modularity allows the efficient implementation of computational experiments in the various applications of category theory where diagrams have become a lingua franca. As an example, we used DisCoPy to perform natural language processing on quantum hardware for the first time.

### Diagrammatic Differentiation for Quantum Machine Learning

https://doi.org/10.4204/EPTCS.343.7

Alexis Toumi, Richie Yeung, Giovanni de Felice

**Abstract:**
We introduce diagrammatic differentiation for tensor calculus by generalising the dual number construction from rigs to monoidal categories. Applying this to ZX diagrams, we show how to calculate diagrammatically the gradient of a linear map with respect to a phase parameter. For diagrams of parametrised quantum circuits, we get the well-known parameter-shift rule at the basis of many variational quantum algorithms. We then extend our method to the automatic differentation of hybrid classical-quantum circuits, using diagrams with bubbles to encode arbitrary non-linear operators. Moreover, diagrammatic differentiation comes with an open-source implementation in DisCoPy, the Python library for monoidal categories. Diagrammatic gradients of classical-quantum circuits can then be simplified using the PyZX library and executed on quantum hardware via the tket compiler. This opens the door to many practical applications harnessing both the structure of string diagrams and the computational power of quantum machine learning.

### DisCoPy for the quantum computer scientist

https://arxiv.org/abs/2205.05190

Alexis Toumi, Giovanni de Felice, Richie Yeung

**Abstract:**
DisCoPy (Distributional Compositional Python) is an open source toolbox for computing with string diagrams and functors. In particular, the diagram data structure allows to encode various kinds of quantum processes, with functors for classical simulation and optimisation, as well as compilation and evaluation on quantum hardware. This includes the ZX calculus and its many variants, the parameterised circuits used in quantum machine learning, but also linear optical quantum computing. We review the recent developments of the library in this direction, making DisCoPy a toolbox for the quantum computer scientist.

### Functorial Language Models

https://arxiv.org/abs/2103.14411

Alexis Toumi, Alex Koziell-Pipe

**Abstract:** We introduce functorial language models: a principled way to compute probability distributions over word sequences given a monoidal functor from grammar to meaning. This yields a method for training categorical compositional distributional (DisCoCat) models on raw text data. We provide a proof-of-concept implementation in DisCoPy, the Python toolbox for monoidal categories.
