---
title: Higher Order Discocat
marimo-version: 0.23.6
---

```python {.marimo}
import marimo as mo
```

# Higher-Order DisCoCat

### (Peirce-Lambek-Montague Semantics)

[arXiv:2311.17813](https://arxiv.org/abs/2311.17813)

## 1) Define Formula as a subclass of frobenius.Diagram

```python {.marimo}
from discopy import frobenius
from discopy.tensor import Dim, Tensor
from discopy.cat import Category, factory

@factory
class Formula(frobenius.Diagram):
    ty_factory = frobenius.PRO  # i.e. natural numbers as objects

    def eval(self, size):
        return frobenius.Functor(
            ob=lambda _: Dim(size),
            ar=lambda box: box.data,
            cod=Category(Dim, Tensor[bool]))(self)

class Cut(frobenius.Bubble, Formula): pass
class Ligature(frobenius.Spider, Formula): pass
class Predicate(frobenius.Box, Formula): pass

Id, Formula.bubble_factory = Formula.id, Cut
Tensor[bool].bubble = lambda self, **_: self.map(lambda x: not x)
```

## 2) Parse natural language sentences using a categorial grammar

```python {.marimo}
from discopy.grammar.categorial import Ty, Word, Eval

n, p, s = Ty('n'), Ty('p'), Ty('s')  # noun, phrase and sentence

Alice = Word("Alice", p)
big, sleeps = Word("big", n << n), Word("sleeps", p >> s)
man, island = (Word(noun, n) for noun in ("man", "island"))
kills, w_is = (Word(verb, (p >> s) << p) for verb in ("kills", "is"))
no, every, some = (Word(det, p << n) for det in ("no", "every", "some"))
```

```python {.marimo}
Alice_kills_a_mortal = (Alice @ kills @ some @ man
    >> p @ ((p >> s) << p) @ Eval(p << n)
    >> p @ Eval((p >> s) << p) >> Eval(p >> s))
Alice_kills_a_mortal.draw(figsize=(6,3))
```

```python {.marimo}
every_big_man_sleeps = (every @ big @ man @ sleeps
    >> ((p << n) @ Eval(n << n) >> Eval(p << n))
    @ (p >> s) >> Eval(p >> s))
every_big_man_sleeps.draw(figsize=(6,3))
```

```python {.marimo}
no_man_is_an_island = (no @ man @ w_is @ some @ island
    >> Eval(p << n) @ ((p >> s) << p) @ Eval(p << n)
    >> p @ Eval((p >> s) << p) >> Eval(p >> s))
no_man_is_an_island.draw(figsize=(6,3))
```

```python {.marimo}
# Generating a random interpretation to test our model

from random import choice

size = 42
random_bits = lambda n=size: [choice([True, False]) for _ in range(n)]

is_killed_by = [random_bits() for _ in range(size)]
unary_predicates = is_Alice, is_man, is_island, is_big, is_sleeping = [
    random_bits() for _ in range(5)]

K = Predicate("K", 1, 1, data=is_killed_by)
A, M, I, B, S = (Predicate(P, 0, 1, data)
                 for P, data in zip("AMIBS", unary_predicates))
```

## 3) Higher-order DisCoCat as a closed functor into Python functions

```python {.marimo}
from typing import Callable

from discopy import closed
from discopy.python import Function

F = closed.Functor(
    cod=Category(tuple[type, ...], Function),
    ob={s: Formula, n: Formula, p: Callable[[Formula], Formula]},
    ar={Alice: lambda: lambda f: A >> f,
        sleeps: lambda: lambda P: P(S.dagger()),
        man: lambda: M, island: lambda: I,
        big: lambda: lambda f: f @ B >> Ligature(2, 1, frobenius.PRO(1)),
        w_is: lambda: lambda P: lambda Q: Q(P(Id(1)).dagger()),
        kills: lambda: lambda P: lambda Q: Q(P(K).dagger()),
        no: lambda: lambda state: lambda effect: (state >> effect).bubble(),
        some: lambda: lambda state: lambda effect: state >> effect,
        every: lambda: lambda state: lambda effect: (
            state >> effect.bubble()).bubble()})

evaluate = lambda sentence: bool(F(sentence)().eval(size))
```

```python {.marimo}
evaluate(Alice_kills_a_mortal), evaluate(every_big_man_sleeps), evaluate(no_man_is_an_island)
```

```python {.marimo}
assert evaluate(Alice_kills_a_mortal) == any(
    is_man[y] and is_killed_by[y][x] and is_Alice[x]
    for x in range(size) for y in range(size))
assert evaluate(every_big_man_sleeps) == all(
    not (is_big[x] and is_man[x]) or is_sleeping[x] for x in range(size))
assert evaluate(no_man_is_an_island) == all(
    not is_man[x] or not is_island[x] for x in range(size))
```

```python {.marimo}
dir(mo.ui)
```

```python {.marimo}
textblock = mo.ui.text("aaa")
textblock
```

```python {.marimo}
textblock.value
```

```python {.marimo}
textblock.value.upper()
```

```python {.marimo}

```