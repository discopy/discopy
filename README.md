
![snake equation](https://raw.githubusercontent.com/oxford-quantum-group/discopy/master/docs/imgs/snake-equation.png)

# Distributional Compositional Python
[![readthedocs](https://readthedocs.org/projects/discopy/badge/?version=master)](https://discopy.readthedocs.io/)
[![Build Status](https://travis-ci.com/oxford-quantum-group/discopy.svg?branch=master)](https://travis-ci.com/oxford-quantum-group/discopy)
[![codecov](https://codecov.io/gh/oxford-quantum-group/discopy/branch/master/graph/badge.svg)](https://codecov.io/gh/oxford-quantum-group/discopy)
[![pylint Score](https://mperlet.github.io/pybadge/badges/9.77.svg)](https://www.pylint.org/)
[![PyPI version](https://badge.fury.io/py/discopy.svg)](https://badge.fury.io/py/discopy)

`discopy` computes natural language meaning in pictures.

## Features

### Diagrams & Recipes

Diagrams are the core data structure of `discopy`, they are generated
by the following grammar:

```python
diagram ::= Box(name, dom=type, cod=type)
    | diagram @ diagram
    | diagram >> diagram
    | Id(type)

type ::= Ty(name) | type.l | type.r | type @ type | Ty()
```

[String diagrams](https://ncatlab.org/nlab/show/string+diagram) (also known as [tensor networks](https://ncatlab.org/nlab/show/tensor+network) or [Penrose notation](https://en.wikipedia.org/wiki/Penrose_graphical_notation)) are a graphical calculus for computing with
[monoidal categories](https://ncatlab.org/nlab/show/monoidal+category).
For example, if we take ingredients as types and cooking steps as boxes then a
diagram is a recipe:

```python
from discopy import Ty, Box, Id

egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
crack = Box('crack', egg, white @ yolk)
swap = lambda x, y: Box('swap', x @ y, y @ x)
merge = lambda x: Box('merge', x @ x, x)

crack_two_eggs = crack @ crack\
    >> Id(white) @ swap(yolk, white) @ Id(yolk)\
    >> merge(white) @ merge(yolk)
crack_two_eggs.draw(path='docs/imgs/crack-eggs.png')
```

![crack two eggs](https://raw.githubusercontent.com/oxford-quantum-group/discopy/master/docs/imgs/crack-eggs.png)

### Snakes & Sentences

The wires in a diagram are never allowed to cross (they are _planar string diagrams_)
but they can be bended using two special kinds of boxes: **cups** and **caps**.
Cups and caps satisfy the **snake equations**, also called [triangle identities](https://ncatlab.org/nlab/show/triangle+identities):

```python
from discopy import Cup, Cap

x = Ty('x')
left_snake = Id(x) @ Cap(x.r, x) >> Cup(x, x.r) @ Id(x)
right_snake =  Cap(x, x.l) @ Id(x) >> Id(x) @ Cup(x.l, x)
assert left_snake.normal_form() == Id(x) == right_snake.normal_form()
```

![snake equations, with types](https://raw.githubusercontent.com/oxford-quantum-group/discopy/master/docs/imgs/typed-snake-equation.png)

In particular, `discopy` can draw the grammatical structure of natural language sentences encoded as reductions in a [pregroup grammar](https://ncatlab.org/nlab/show/pregroup+grammar) (see Lambek, [From Word To Sentence (2008)](http://www.math.mcgill.ca/barr/lambek/pdffiles/2008lambek.pdf) for an  introduction).

```python
from discopy import pregroup, Word

s, n = Ty('s'), Ty('n')
Alice, Bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r @ s @ n.l)

sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
pregroup.draw(sentence, path='docs/imgs/alice-loves-bob.png')
```

![Alice loves Bob](https://raw.githubusercontent.com/oxford-quantum-group/discopy/master/docs/imgs/alice-loves-bob.png)

### Functors & Meanings

*Functors* compute the meaning of a diagram, given a meaning for each box.
Free functors (i.e. from diagrams to diagrams) can fill each box with a complex diagram, the result can then be simplified using `diagram.normalize()` to remove the snakes.

```python
love_box = Box('loves', n @ n, s)
love_ansatz = Cap(n.r, n) @ Cap(n, n.l) >> Id(n.r) @ love_box @ Id(n.l)
ob, ar = {s: s, n: n}, {Alice: Alice, Bob: Bob, loves: love_ansatz}
A = RigidFunctor(ob, ar)
rewriting_steps = A(sentence).normalize()
Diagram.to_gif(A(sentence), diagrams=rewriting_steps
               path='docs/imgs/autonomisation.gif')
```

![autonomisation](https://raw.githubusercontent.com/oxford-quantum-group/discopy/master/docs/imgs/autonomisation.gif)

Functors into the category of matrices evaluate a diagram as a tensor network
using [numpy](https://numpy.org/).
Applied to pregroup diagrams, matrix functors implement the
**distributional compositional** (_DisCo_) models of [Clark, Coecke, Sadrzadeh (2008)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.363.8703&rep=rep1&type=pdf).

```python
import numpy as np
from discopy import MatrixFunctor

ob = {s: 1, n: 2}
ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
F = MatrixFunctor(ob, ar)

assert F(sentence) == np.array([1])
```

## Getting Started

```shell
pip install discopy
```

## Documentation

The documentation is hosted at [readthedocs.io](https://discopy.readthedocs.io/),
you can also checkout the [notebooks](notebooks/) for a demo!
