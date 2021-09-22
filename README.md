
![snake equation](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/snake-equation.png)

# Distributional Compositional Python

DisCoPy is a tool box for computing with monoidal categories.

## Features

### Diagrams & Recipes

Diagrams are the core data structure of DisCoPy, they are generated
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
from discopy import Ty, Box, Id, Swap

egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
crack = Box('crack', egg, white @ yolk)
merge = lambda x: Box('merge', x @ x, x)

crack_two_eggs = crack @ crack\
    >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
    >> merge(white) @ merge(yolk)
crack_two_eggs.draw(path='docs/_static/imgs/crack-eggs.png')
```

![crack two eggs](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/crack-eggs.png)

### Snakes & Sentences

Wires can be bended using two special kinds of boxes: **cups** and **caps**, which satisfy the **snake equations**, also called [triangle identities](https://ncatlab.org/nlab/show/triangle+identities).

```python
from discopy import Cup, Cap

x = Ty('x')
left_snake = Id(x) @ Cap(x.r, x) >> Cup(x, x.r) @ Id(x)
right_snake =  Cap(x, x.l) @ Id(x) >> Id(x) @ Cup(x.l, x)
assert left_snake.normal_form() == Id(x) == right_snake.normal_form()
```

![snake equations, with types](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/typed-snake-equation.png)

In particular, DisCoPy can draw the grammatical structure of natural language sentences encoded as reductions in a [pregroup grammar](https://ncatlab.org/nlab/show/pregroup+grammar) (see Lambek, [From Word To Sentence (2008)](http://www.math.mcgill.ca/barr/lambek/pdffiles/2008lambek.pdf) for an  introduction).

```python
from discopy import grammar, Word

s, n = Ty('s'), Ty('n')
Alice, Bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r @ s @ n.l)

sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
grammar.draw(sentence,
             path='docs/_static/imgs/alice-loves-bob.png',
             fontsize=20, fontsize_types=12)
```

![Alice loves Bob](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/alice-loves-bob.png)

### Functors & Rewrites

**Monoidal functors** compute the meaning of a diagram, given an interpretation for each wire and for each box.
In particular, **tensor functors** evaluate a diagram as a tensor network using [numpy](https://numpy.org/).
Applied to pregroup diagrams, DisCoPy implements the
**distributional compositional** (_DisCo_) models of
[Clark, Coecke, Sadrzadeh (2008)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.363.8703&rep=rep1&type=pdf).

```python
from discopy import TensorFunctor

F = TensorFunctor(
    ob={s: 1, n: 2},
    ar={Alice: [1, 0], loves: [[0, 1], [1, 0]], Bob: [0, 1]})

assert F(sentence) == 1
```

**Free functors** (i.e. from diagrams to diagrams) can fill each box with a complex diagram.
The result can then be simplified using `diagram.normalize()` to remove the snakes.

```python
from discopy import Functor

def wiring(word):
    if word.cod == n:  # word is a noun
        return word
    if word.cod == n.r @ s @ n.l:  # word is a transitive verb
        return Cap(n.r, n) @ Cap(n, n.l)\
            >> Id(n.r) @ Box(word.name, n @ n, s) @ Id(n.l)

W = Functor(ob={s: s, n: n}, ar=wiring)


rewrite_steps = W(sentence).normalize()
sentence.to_gif(*rewrite_steps, path='autonomisation.gif', timestep=1000)
```

![autonomisation](docs/_static/imgs/autonomisation.gif)


### Loading Corpora
You can load "Alice in Wonderland" in DisCoCat form with a single command:
```python
from discopy import utils
url = "https://qnlp.cambridgequantum.com/corpora/alice/discocat.zip"
diagrams = utils.load_corpus(url)
```
Find more DisCoCat resources at https://qnlp.cambridgequantum.com/downloads.html.

## Getting Started

```shell
pip install discopy
```

## Contributing

Contributions are welcome, please drop one of us an email or
[open an issue](https://github.com/oxford-quantum-group/discopy/issues/new).

## Tests

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

The documentation is built automatically from the source code using
[sphinx](https://www.sphinx-doc.org/en/master/).
If you need to build it locally, just run:

```shell
(cd docs && (make clean; make html))
```

## Documentation

The tool paper is now available on [arXiv:2005.02975](https://arxiv.org/abs/2005.02975), it was presented at [ACT2020](https://act2020.mit.edu/).

The documentation is hosted at [readthedocs.io](https://discopy.readthedocs.io/),
you can also checkout the [notebooks](https://discopy.readthedocs.io/en/main/notebooks.html) for a demo!

[![readthedocs](https://readthedocs.org/projects/discopy/badge/?version=main)](https://discopy.readthedocs.io/)
[![Build Status](https://travis-ci.com/oxford-quantum-group/discopy.svg?branch=main)](https://travis-ci.com/oxford-quantum-group/discopy)
[![codecov](https://codecov.io/gh/oxford-quantum-group/discopy/branch/main/graph/badge.svg)](https://codecov.io/gh/oxford-quantum-group/discopy)
[![PyPI version](https://badge.fury.io/py/discopy.svg)](https://badge.fury.io/py/discopy)
[![arXiv:2005.02975](http://img.shields.io/badge/math.CT-arXiv%3A2005.02975-brightgreen.svg)](https://arxiv.org/abs/2005.02975)
