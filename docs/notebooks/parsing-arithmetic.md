---
title: Parsing Arithmetic
marimo-version: 0.23.14
---

```python {.marimo}
import marimo as mo
```

# Parsing arithmetic with proof nets

The sequence `a + b * c` has two readings, `a + (b * c)` and `(a + b) * c`, and the convention that `*` binds tighter than `+` picks the first one.
In this notebook we teach a neural network that convention from examples, with nothing but the modules
[`grammar.proofnet`](https://docs.discopy.org/en/main/_api/discopy.grammar.proofnet.html) and
[`grammar.neural`](https://docs.discopy.org/en/main/_api/discopy.grammar.neural.html):
the network never sees a bracket, it learns to link the operands of each operator.

## Formulae are terms

An arithmetic formula is a lambda term over one atomic type `e` for expressions: a variable is a constant of type `e` and an operator a constant of type `e >> (e >> e)`, taking its left operand then its right one.
Each token of a formula is a word, and its position in the formula is the `data` of its constant, so that the two `+` of `a + b + c` are two different words.

```python {.marimo}
import random
from itertools import permutations

import torch
from torch import nn

from discopy.grammar.abstract import Ty, Constant, Functor
from discopy.grammar.proofnet import ProofNet
from discopy.grammar.neural import Signature, Parser
from discopy.python import Function
from discopy.utils import AxiomError

e = Ty("e")
operator = e >> (e >> e)


def tokenise(formula):
    return [Constant(token, operator if token in "+*" else e, data=i)
            for i, token in enumerate(formula.split())]


a, plus, b, times, c = tokenise("a + b * c")
readings = plus(a)(times(b)(c)), times(plus(a)(b))(c)
mo.md(f"`{readings[0]}`")
```

To read a term back as a formula, we evaluate it with a functor into Python functions: `e` goes to `str`, a variable to its name and an operator to the function writing it between its two operands, with brackets around the compound ones.

```python {.marimo}
def bracket(operand):
    return f"({operand})" if " " in operand else operand


show = Functor(
    ob_map={e: str},
    ar_map=lambda word: (lambda: word.name) if word.cod == e else (
        lambda: lambda x: lambda y: f"{bracket(x)} {word.name} {bracket(y)}"),
    cod=Function)

[show(term)() for term in readings]
```

## Two readings, one sequent

Both readings use the same words with the same types: they are two proofs of the same sequent `a : e, + : e >> (e >> e), b : e, * : e >> (e >> e), c : e |- e`.
What tells them apart is their axiom linking, which wanted port of each operator is linked to which given port.
Each type has its ports, `e` one given port and `e >> (e >> e)` two wanted ones then a given one, and the goal `e` is wanted.

```python {.marimo}
nets = [ProofNet.from_term(term, (a, plus, b, times, c)) for term in readings]
sequent = nets[0].sequent
mo.md("\n".join(
    [f"| port | {' | '.join(map(str, range(len(sequent.ports))))} |",
     "|---" * (len(sequent.ports) + 1) + "|",
     "| polarity | " + " | ".join(
         "given" if given else "wanted" for _, given in sequent.ports) + " |"]
    + [f"| `{show(net.to_term())()}` | " + " | ".join(
        str(dict(net.links + tuple(map(reversed, net.links)))[i])
        for i in range(len(sequent.ports))) + " |" for net in nets]))
```

Each row of the table says which port is linked to which.
Since every port has the same atom, any bijection between the five wanted ports and the five given ones is a linking, there are $5! = 120$ of them.
Most are not proofs: following the links from the goal, `ProofNet.to_term` fails as soon as a word cannot be reached or an operator would feed its own operand.
The ones that are correct are exactly the terms that use each word once, with their operands in any order:

```python {.marimo}
(negatives, positives), = sequent.blocks.values()


def is_correct(net):
    try:
        net.to_term()
        return True
    except AxiomError:
        return False


linkings = [ProofNet(sequent, zip(negatives, permutation))
            for permutation in permutations(positives)]
correct = [show(net.to_term())() for net in linkings if is_correct(net)]
mo.md(f"{len(correct)} correct linkings out of {len(linkings)}: "
      + ", ".join(f"`{formula}`" for formula in correct))
```

Only two of these keep the words in order, and only one of those follows the convention.
Parsing is picking it, which is what we are going to learn.

## Gold parses

The convention is what precedence climbing implements: the training data is a list of random formulae, each with the proof net of its conventional reading.
We train on formulae of up to four operators and test on longer ones, of five and six operators, which the parser has never seen.

```python {.marimo}
PRECEDENCE = {"+": 1, "*": 2}


def climb(words, level=1):
    term = words.pop(0)
    while words and PRECEDENCE[words[0].name] >= level:
        head = words.pop(0)
        term = head(term)(climb(words, PRECEDENCE[head.name] + 1))
    return term


def gold(formula):
    words = tokenise(formula)
    return ProofNet.from_term(climb(list(words)), words)


rng = random.Random(0)


def random_formula(n_operators):
    return " ".join([rng.choice("abcd")] + [
        token for _ in range(n_operators)
        for token in (rng.choice("+*"), rng.choice("abcd"))])


train = [random_formula(rng.randint(1, 4)) for _ in range(2000)]
test = [random_formula(rng.randint(5, 6)) for _ in range(200)]
mo.md(f"`{train[0]}` is parsed as `{show(gold(train[0]).to_term())()}`.")
```

## Encoder and parser

A parser reads the states of the words given by any encoder.
Ours embeds each token and runs a bidirectional GRU over the formula, so that the state of each word knows what surrounds it: an `a` before a `*` is not an `a` before a `+`.
The `Parser` then does two things with these states: its tagger learns to write the type of each word symbol by symbol, here only ever `e` or `e >> (e >> e)`, and its linker learns to score the pairs of ports, so that the Hungarian algorithm can pick the best linking.

```python {.marimo}
VOCABULARY = "abcd+*"


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCABULARY), dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)

    def forward(self, formulae):
        tokens = [formula.split() for formula in formulae]
        ids = torch.zeros(len(tokens), max(map(len, tokens)), dtype=torch.long)
        for i, sentence in enumerate(tokens):
            ids[i, :len(sentence)] = torch.tensor(
                [VOCABULARY.index(token) for token in sentence])
        mask = torch.arange(ids.shape[1])[None] < torch.tensor(
            [len(sentence) for sentence in tokens])[:, None]
        return self.gru(self.embedding(ids))[0], mask


torch.manual_seed(0)
encoder = Encoder(dim=16)
parser = Parser(Signature((e, )), hidden=32, dim=32)
optimiser = torch.optim.Adam(
    [*parser.parameters(), *encoder.parameters()], lr=3e-3)
gold_nets = {formula: gold(formula) for formula in train}
losses = []
for _ in range(400):
    batch = rng.sample(train, 32)
    optimiser.zero_grad()
    loss = parser.loss(*encoder(batch), [gold_nets[formula] for formula in batch])
    loss.backward()
    optimiser.step()
    losses.append(loss.item())
encoder, parser = encoder.eval(), parser.eval()
```

The loss is the sum of that of the tagger and that of the linker, the likelihood of the gold linking after Sinkhorn normalisation.

```python {.marimo}
import matplotlib.pyplot as plt

figure, axes = plt.subplots(figsize=(6, 3))
axes.plot(losses)
axes.set(xlabel="step", ylabel="loss", yscale="log")
figure
```

## Parsing

Parsing a formula is one call: the tagger writes the type of each word, the linker scores the ports and the Hungarian algorithm links them.
The proof net it returns is read back as a term by `to_term`, which fails if the linking is not correct.

```python {.marimo}
def parse(formula):
    with torch.no_grad():
        states, _ = encoder([formula])
        net = parser(formula.split(), states[0], e)
    return show(net.to_term())()


examples = ["a + b * c", "a * b + c", "a + b + c", "a * b * c + d * a"]
mo.md("\n".join([
    "| formula | parse |", "|---|---|"]
    + [f"| `{formula}` | `{parse(formula)}` |" for formula in examples]))
```

On the formulae of five and six operators held out for testing, longer than any the parser was trained on:

```python {.marimo}
def is_right(formula):
    try:
        return parse(formula) == show(gold(formula).to_term())()
    except AxiomError:
        return False


accuracy = sum(map(is_right, test)) / len(test)
mo.md(f"{accuracy:.0%} of the {len(test)} test formulae are parsed right.")
```

Finally, the parse of `a + b * c` as a diagram, the term evaluated in the free closed category: the words are boxes and each application of an operator to an operand is an evaluation, drawn as a box `__call__`.

```python {.marimo}
with torch.no_grad():
    abc, _ = encoder(["a + b * c"])
    parsed = parser("a + b * c".split(), abc[0], e).to_term()
parsed.eval()
```
