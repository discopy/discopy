---
title: Examples
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Examples

Taken from _DisCoPy: the Hierarchy of Graphical Languages in Python_ ([arXiv:2311.10608](https://arxiv.org/abs/2311.10608))
<!---->
## A diagram as a list of layers

```python {.marimo}
from discopy.monoidal import Ty, Box, Layer, Diagram

x, y, z = Ty('x'), Ty('y'), Ty('z')
f, g, h = Box('f', x, y @ z), Box('g', y, z), Box('h', z, z)

assert f >> g @ h == Diagram(
    dom=x, cod=z @ z, inside=(
        Layer(Ty(), f, Ty()),
        Layer(Ty(), g, z),
        Layer(z,    h, Ty())))

(f >> g @ h).draw()
```

## Boolean circuits as a subclass of Diagram

```python {.marimo}
from discopy import monoidal, python
from discopy.cat import ar_factory

@ar_factory  # Ensure that composition of circuits remains a circuit.
class Circuit(monoidal.Diagram):
    ob = monoidal.PRO  # Use natural numbers as objects.

    def __call__(self, *bits):
        F = monoidal.Functor(
            ob=lambda _: bool, ar=lambda f: f.data,
            cod=python.Function)
        return F(self)(*bits)

class Gate(monoidal.Box, Circuit):
    """A gate is just a box in a circuit with a function as data."""

NAND = Gate("NAND", 2, 1, data=lambda x, y: not (x and y))
COPY = Gate("COPY", 1, 2, data=lambda x: (x, x))

XOR = COPY @ COPY >> 1 @ (NAND >> COPY) @ 1 >> NAND @ NAND >> NAND
CNOT = COPY @ 1 >> 1 @ XOR
NOTC = 1 @ COPY >> XOR @ 1
SWAP = CNOT >> NOTC >> CNOT  # Exercise: Find a cheaper SWAP circuit!

assert all(SWAP(x, y) == (y, x) for x in [True, False]
                                for y in [True, False])

SWAP.draw(figsize=(4, 8), wire_labels=False)
```

## Spirals as the worst-case for normalisation

```python {.marimo}
from discopy.drawing import Equation
x_1 = Ty('x')
f_1, u = (Box('f', Ty(), x_1 @ x_1), Box('u', Ty(), x_1))

def spiral(length):
    diagram, n = (u, length // 2 - 1)
    for i in range(n):
        diagram = diagram >> x_1 ** i @ f_1 @ x_1 ** (i + 1)
    diagram = diagram >> x_1 ** n @ u.dagger() @ x_1 ** n
    for i in range(n):
        m = n - i - 1
        diagram = diagram >> x_1 ** m @ f_1.dagger() @ x_1 ** m
    return diagram
assert spiral(8).dagger() != spiral(8)
assert spiral(8).dagger() == spiral(8).normal_form()
Equation(spiral(8), spiral(8).dagger()).draw()
```

## The golden ratio as the trace of a string diagram

```python {.marimo}
from discopy import traced
x_2 = traced.Ty('x')
add, div, one = (traced.Box('+', x_2 @ x_2, x_2), traced.Box('/', x_2 @ x_2, x_2), traced.Box('1', traced.Ty(), x_2))
copy = traced.Box('', x_2, x_2 @ x_2, draw_as_spider=True, color='black')
phi = ((one >> copy) @ x_2 >> x_2 @ div >> add >> copy).trace()
F = traced.Functor(ob={x_2: float}, ar={div: lambda x, y=1.0: x / y, add: lambda x, y: x + y, copy: lambda x: (x, x), one: lambda: 1.0}, cod=python.Function)
with python.Function.no_type_checking:
    assert F(phi)() == 0.5 * (1 + 5 ** 0.5)
# The default y=1 is the initial value for the fixed point.
phi.draw()
```

## The Kauffman bracket as a ribbon functor

```python {.marimo}
from discopy import ribbon, drawing
x_3 = ribbon.Ty('x')
cup, cap, braid = (ribbon.Cup(x_3.r, x_3), ribbon.Cap(x_3.r, x_3), ribbon.Braid(x_3, x_3))
link = cap >> x_3.r @ cap @ x_3 >> braid.r @ braid >> x_3.r @ cup @ x_3 >> cup

@ar_factory
class Kauffman(ribbon.Diagram):
    ob = ribbon.PRO

class Cup(ribbon.Cup, Kauffman):
    pass

class Cap(ribbon.Cap, Kauffman):
    pass

class Sum(ribbon.Sum, Kauffman):
    pass
Kauffman.cup_factory = Cup
Kauffman.cap_factory = Cap
Kauffman.sum_factory = Sum

class Variable(ribbon.Box, Kauffman):
    pass
Kauffman.braid = lambda x, y: Variable('A', 0, 0) @ x @ y + (Cup(x, y) >> Variable('A', 0, 0).dagger() >> Cap(x, y))
K = ribbon.Functor(ob=lambda _: 1, ar={}, cod=Kauffman)
drawing.Equation(link, K(link), symbol='$\\mapsto$').draw(figsize=(8, 4), wire_labels=False)
```

## Checking the equality of two diagrams

```python {.marimo}
from discopy import symmetric
x_4, y_1, z_1 = (symmetric.Ty('x'), symmetric.Ty('y'), symmetric.Ty('z'))
f_2 = symmetric.Box('f', x_4, y_1 @ z_1)
g_1, h_1 = (symmetric.Box('g', z_1, x_4), symmetric.Box('h', y_1, z_1))
diagram_left = f_2 >> symmetric.Swap(y_1, z_1) >> g_1 @ h_1
diagram_right = f_2 >> h_1 @ g_1 >> symmetric.Swap(z_1, x_4)
assert diagram_left != diagram_right
with symmetric.Diagram.hypergraph_equality:
    assert diagram_left == diagram_right
drawing.Equation(diagram_left, diagram_right).draw()
```

## Defining a diagram from a Python function

```python {.marimo}
from discopy import markov

mx, my = markov.Ty('x'), markov.Ty('y')
mf = markov.Box('f', mx @ mx, my)

@markov.Diagram.from_callable(mx @ mx, my)
def markov_diagram(a, b):  # Take two wires as inputs
    _ = mf(b, a)     # Swap, apply f and discard the result.
    return mf(a, b)  # Apply f again and return the result.

assert markov_diagram == markov.Copy(mx) @ markov.Copy(mx)\
    >> mx @ (markov.Swap(mx, mx) >> mf >> markov.Discard(my)) @ mx >> mf

markov_diagram.draw()
```

## The hypergraph representation of a diagram

```python {.marimo}
from discopy import frobenius

fx, fy = frobenius.Ty('x'), frobenius.Ty('y')
ff, fg = frobenius.Box('f', fx, fy), frobenius.Box('g', fy @ fy, fx)

diagram_lhs = frobenius.Swap(fy, fx) >> fx @ frobenius.Cap(fx, fx) @ fy >> frobenius.Spider(2, 2, fx) @ ff @ fy >> fx @ fx @ fg\
    >> fx @ frobenius.Cup(fx, fx) @ frobenius.Spider(0, 0, fx)

diagram_rhs = frobenius.Cap(fy, fy) @ fy @ fx >> fy @ fg @ fx >> fy @ frobenius.Spider(2, 2, fx) @ frobenius.Cap(fx, fx)\
    >> fy @ ff @ fx @ frobenius.Cup(fx, fx) >> frobenius.Cup(fy, fy) @ fx

fa, fb, fc, fd = "abcd"
hypergraph = frobenius.Hypergraph(
    dom=fy @ fx, cod=fx, boxes=(ff, fg),
    wires=((fc, fa),           # input wires of the hypergraph
           (((fa, ), (fb, )),    # input and output wires of f
            ((fb, fc), (fa, ))),  # input and output wires of g
           (fa, )),           # output wire of the hypergraph
    spider_types={fa: fx, fb: fy, fc: fy, fd: fx})  # note the extra x

assert diagram_lhs.to_hypergraph() == hypergraph == diagram_rhs.to_hypergraph()

drawing.Equation(diagram_lhs, diagram_rhs).draw()
```

## First-order logic with diagrams

As pioneered by C.S. Peirce: boxes are predicates, spiders are variables and bubbles are negation.

```python {.marimo}
from discopy.tensor import Dim, Tensor

Tensor[bool].bubble = lambda self, **_: self.map(lambda x: not x)

@ar_factory
class Formula(frobenius.Diagram):
    ob = frobenius.PRO

    def eval(self, size, model):
        return frobenius.Functor(
            ob=lambda _: Dim(size), ar=lambda f: model[f],
            cod=Tensor[bool])(self)

class Cut(frobenius.Bubble, Formula): pass
class Ligature(frobenius.Spider, Formula): pass
class Predicate(frobenius.Box, Formula): pass

P = Predicate("P", 0, 2)  # A binary predicate, i.e. a relation.
G, M = [Predicate(unary, 0, 1) for unary in ("G", "M")]
lp, lg, lm = [[0, 1], [0, 0]], [0, 1], [1, 0]
size, model = 2, {G: lg, M: lm, P: lp}

formula = G >> Ligature(1, 2, frobenius.PRO(1))\
    >> Cut(Cut(Formula.id(1)) >> G.dagger())\
    @ (M @ 1 >> P.dagger())

assert bool(formula.eval(size, model)) == any(
    lg[x] and all(not lg[y] or x == y for y in range(size))
    and lm[z] and lp[z][x] for x in range(size) for z in range(size))

formula.draw(wire_labels=False)
```