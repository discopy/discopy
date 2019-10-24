# discopy
Distributional Compositional Python

## Requirements

* numpy 1.17.2
* pytket (only for the circuit module)

## Example

```python
import numpy as np
from disco import Word, Parse, Model, s, n, l, r

alice, bob = Word('Alice', [n]), Word('Bob', [n])
loves = Word('loves', [l(*n), s, r(*n)])
sentence = Parse([alice, loves, bob], [0, 1])

F = Model(['Alice', 'loves', 'Bob'], {s: 1, n: 2},
          {alice : np.array([0, 1]),
           bob : np.array([1, 0]),
           loves : np.array([[0, 1], [1, 0]])})

assert F(sentence)
```
