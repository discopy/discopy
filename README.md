# discopy
Distributional Compositional Python

![Alice loves Bob.](figures/alice-loves-bob.png)

## Requirements

* `numpy`
* `pytket`

## Example

```python
from numpy import array
from disco import Type, Word, Parse, Model

s, n = Type('s'), Type('n')
alice, bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r + s + n.l)
sentence = Parse([alice, loves, bob], [0, 1])

F = Model({s: 1, n: 2},
    {
        alice: array([1, 0]),
        loves: array([0, 1, 1, 0]),
        bob: array([0, 1])
    })

assert F(sentence)
```
