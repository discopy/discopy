---
theme: gaia
_class: lead
backgroundColor: #000
---

<style>
* { text-align: left; color: white; }
h1, strong { color: darkred; }
a { color: orange; }
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    background-color: white;
}
</style>

<style scoped> * { text-align: center; }</style>

# Category Theory for Quantum Natural Language Processing

$$\quad$$

Alexis TOUMI

15 Novembre 2022, Paris

---

<style scoped> * { text-align: center; }</style>

# ~~Category Theory for Quantum Natural Language Processing~~
# DisCoPy: a toolkit for computing with string diagrams

$$\quad$$

Alexis TOUMI

15 Novembre 2022, Paris

---

# QNLP: the recipe

### Tree ingredients

$\text{Natural Language}
\xrightarrow{\text{Category Theory}}
\text{Quantum Computing}$

$$\quad$$

### Three steps

1) **Parse** the given text to get a *string diagram*
2) **Map** the grammar to a circuit using a *functor*
3) **Tune** the parameters to solve a data-driven task

---

# You already use string diagrams without knowing it

* Quantum circuits (see **ZX-calculus**)
* Tensor networks
* Neural networks
* Concurrent processes
* Electrical circuits
* Logical formulae (see **existential graphs**)
* The grammar of this sentence is a string diagram!

---

# So what is a string diagram?

A **box** represents any process with a list of **wires** as input and output.

A **signature** is a collection of boxes and wires.

String diagrams can be defined by recursion:

* every box $f : x \to y$ is also a string diagram,
* the **identity** $\text{id}(x) : x \to x$ on a list of wires $x$ is a string diagram,
* so is the **composition** $g \circ f : x \to z$ of $f : x \to y$ and $g : y \to z$
* and the **tensor** $f \otimes f' : xx' \to yy'$ of $f : x \to y$ and $f' : x' \to y'$.

---

# Cooking recipes are string diagrams!

![height:400px](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/crack-eggs.png)

* Pawel Sobocinski's **Graphical Linear Algebra**, i.e. linear algebra with all string diagrams and no vectors!

---

# This sentence is a string diagram

![height:400px](https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Cgisf-tgg.svg/1280px-Cgisf-tgg.svg.png)

- Noam Chomsky, **Syntactic Structures** (1957)

---

# This sentence is a string diagram

![height:350px](https://cqc.pythonanywhere.com/discocat/png?sentence=This%20sentence%20is%20a%20string%20diagram&size=small)

- Joachim Lambek, **The mathematics of sentence structure** (1958)

* Joachim Lambek, **Type grammar revisited** (1997)

---

# This Python code is a string diagram

```python
from discopy import Ty, Word, Id, Cup

s, n = Ty('s'), Ty('n')
Alice, loves, Bob = Word('Alice', n), Word('loves', n.r @ s @ n.l), Word('Bob', n)

sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
```

![height:300px](https://cqc.pythonanywhere.com/discocat/png)

---

# Monoidal categories

A monoidal category $C$ is just a pair of types for **objects** and **arrows** with methods:

- `dom` and `cod` from arrows to objects,
- `id` from objects to arrows,
- `then` and `tensor` from pairs of arrows to arrows.

For example:

- `(Ty, Diagram)` is the **free monoidal category** on a given signature,
- `(type, Function)`, `(int, Matrix)`, `(list[int], Tensor)`, etc.

---

# Monoidal functors

A monoidal functor $F : C \to D$ is just a pair of mappings:

- `ob` from objects of $C$ to objects of $D$,
- `ar` from arrows of $C$ to arrows of $D$.

For example $F :$ `(Ty, Diagram)` $\to$ `(list[int], Tensor)`

```python
from discopy.tensor import Functor

F = Functor(ob={s: 1, n: 2},
            ar={Alice: [1, 0], loves: [[0, 1], [1, 0]], Bob: [0, 1]})

assert F(sentence)
```

---

# QNLP models

We define a QNLP model as a monoidal functor

$$
F : \mathbf{Grammar} \to \mathbf{Circuit}
$$

![](https://discopy.readthedocs.io/en/main/_images/functor-example2.png)

---

# QNLP models

We define a QNLP model as a monoidal functor

$$
F : \mathbf{Grammar} \to \mathbf{Circuit}
$$

```python
from discopy.quantum import qubit, Ket, H, X, CX, sqrt
from discopy.circuit import Functor

F_ = circuit.Functor(
    ob={s: Ty(), n: qubit},
    ar={Alice: Ket(0), loves: sqrt(2) @ Ket(0, 0) >> H @ X >> CX, Bob: Ket(1)})

assert F_(sentence).eval() == F(sentence)
```

---

# Diagrammatic Differentiation for Quantum Machine Learning

joint work with **Giovanni de Felice** and **Richie Yeung**

![height:400px](https://discopy.readthedocs.io/en/main/_images/notebooks_diag-diff_11_0.png)

---

# Check it out!

![width:800px](https://raw.githubusercontent.com/oxford-quantum-group/discopy/main/docs/_static/imgs/snake-equation.png)

- **https://github.com/discopy/discopy**
- **https://discopy.readthedocs.io**
- **https://discopy.org**
