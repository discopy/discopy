# Distributional Compositional Python

`discopy` computes meaning in pictures.

!["Alice loves Bob" in picture](figures/alice-loves-bob.png)

## Natural Language Meaning

The recipe goes in three steps:

1) draw the picture

```python
from disco import Type, Word, Cup, Wire

s, n = Type('s'), Type('n')

alice = Word('Alice', n)
loves = Word('loves', n.r + s + n.l)
bob = Word('Bob', n)

words = alice.tensor(loves).tensor(bob)
grammar = Cup(n).tensor(Wire(s)).tensor(Cup(n.l))
sentence = words.then(grammar)
```

2) fill in the picture with `numpy` arrays

```python
from disco import Model
from numpy import array

F = Model({s: 1, n: 2},
    {
        alice: array([1, 0]),
        loves: array([0, 1, 1, 0]),
        bob: array([0, 1])
    })
```

3) compute the meaning!

```python
assert F(sentence) == True
```

## General Abstract Nonsense

`discopy` is a Python implementation of the categorical compositional categorical (DisCoCat) models, see [arXiv:1003.4394](https://arxiv.org/abs/1003.4394), [arXiv:1106.4058](https://arxiv.org/abs/1106.4058) [arXiv:1904.03478](https://arxiv.org/abs/1904.03478).

* `cat.Arrow`, `cat.Identity`, `cat.Generator` implement free categories.
* `cat.Functor` implements Python-valued functors.
* `moncat.Diagram`, `moncat.Wire`, `moncat.Node` implement free monoidal categories.
* `moncat.MonoidalFunctor` implements free monoidal functors.
* `disco.NumpyFunctor` implements monoidal functors into the category of matrices.
* `disco.Word`, `disco.Cup`, `disco.Parse` implement pregroup grammars.
