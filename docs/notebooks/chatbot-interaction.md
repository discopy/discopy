
# Geometry of Chatbot Interaction

This used to be the second half of DisCoPy [README](../../README.md).

## From states to processes

The [`Int`](https://docs.discopy.org/en/main/_api/discopy.interaction.Int.html)-construction of [Joyal, Street & Verity (1996)](https://doi.org/10.1017/S0305004100074338) is

> a glorification of the construction of the integers from the natural numbers

i.e. the same way we can freely add inverses to a commutative monoid to get a group, e.g. $\mathbb{N} \hookrightarrow Int(\mathbb{N}) = \mathbb{Z}$ where

$$
Int(M) \ = \ (M \times M) \ / \ \set{ (x, x') \sim (y, y') \ \vert \ x + y' = x' + y }
$$

you can freely add cups and caps to a [`symmetric`](https://docs.discopy.org/en/main/_api/discopy.symmetric.html) or [`balanced`](https://docs.discopy.org/en/main/_api/discopy.balanced.html) category to get a [`compact`](https://docs.discopy.org/en/main/_api/discopy.compact.html) or [`tortile`](https://docs.discopy.org/en/main/_api/discopy.tortile.html) category.

The only condition is that the monoid needs to be **cancellative**, i.e. $x + n = y + n \implies x = y$.

The [vertical categorification](https://ncatlab.org/nlab/show/vertical+categorification) of a cancellative monoid is called a [`traced`](https://docs.discopy.org/en/main/_api/discopy.traced.html) category, where the diagrams can have feedback loops:

![right trace](https://github.com/discopy/discopy/blob/97c002fa8eaefefc53287d960a54ebd5ac96dedd/docs/_static/traced/right-trace.png)

Given a traced category $C$, we construct $Int(C)$ with objects given by pairs of objects $Ob(Int(C)) = Ob(C) \times Ob(C)$, arrows given by $Int(C)((x_0, x_1), (y_0, y_1)) = C(x_0 \otimes y_1, x_1 \otimes y_0)$ and the composition is given by **symmetric feedback**:

![symmetric feedback](https://github.com/discopy/discopy/blob/97c002fa8eaefefc53287d960a54ebd5ac96dedd/docs/_static/int/symmetric-feedback.png)

The structure theorem of Joyal-Street-Verity says that the embedding $C \hookrightarrow Int(C)$ is fully-faithful, i.e. we can remove all the snakes and replace all the cups and caps with feedback loops.
We can use this geometry of interaction to interpret words as processes rather than states:

```python
from discopy.interaction import Ty, Int
from discopy.compact import Ty as T, Diagram as D, Box

N, S = T('N'), T('S')
A, B = Box('A', N, N), Box('B', N, N)
L = Box('L', N @ S @ N, N @ S @ N)
swaps = D.permutation((2, 1, 0), N @ S @ N)
G = pregroup.Functor(
    ob={s: Ty[T](S, S), n: Ty[T](N, N)},
    ar={Alice: A, loves: swaps >> L, Bob: B},
    cod=Int(D))

ALB_trace = (A @ S @ B >> L).trace(left=True).trace(left=False).foliation()

with D.hypergraph_equality:
  assert G(sentence).inside == ALB_trace

Equation(sentence.foliation(), ALB_trace, symbol="$\\mapsto$").draw()
```

![Alice loves traces](https://github.com/discopy/discopy/raw/main/docs/_static/int/alice-loves-traces.png)

### Streams and delayed feedback

A key axiom of traced monoidal categories which allows to simplify diagrams is the **yanking equation**:

![yanking](https://github.com/discopy/discopy/raw/main/docs/_static/traced/yanking.png)

If we relax this assumption we get the concept of a [`feedback`](https://docs.discopy.org/en/main/_api/discopy.feedback.html) category where the objects come with a [`delay`](https://docs.discopy.org/en/main/_api/discopy.feedback.Ob.html#discopy.feedback.Ob.delay) operation and the feedback loops have a more restricted shape:

![feedback operator](https://github.com/discopy/discopy/raw/main/docs/_static/feedback/feedback-operator.png)

Given a symmetric category $C$, we can construct a feedback category of **monoidal streams** $Stream(C)$ where

- the objects are infinite sequences of objects $Ob(Stream(C)) = C \times Ob(Stream(C))$,
- the arrows are infinite sequences of arrows $Stream(C)(X, Y) = \coprod_{M} Stream(C)(X, Y, M)$ defined by:

$$Stream(C)(X, Y, M) = C(X_0 \otimes M_0, Y_0 \otimes M_1)  \times Stream(C)(X^+, Y^+, M^+)$$

where $X_0$ and $X^+$ are the head and the tail of the stream $X$.

This comes with a delay $d(X) \in Ob(Stream(C))$ given by the monoidal unit as head $d(X)_0 = I$ and the given object as tail $d(X)^+ = X$.
The feedback operation is given by:

![feedback unrolling](https://github.com/discopy/discopy/raw/main/docs/_static/stream/feedback-unrolling.png)

We can use this to unroll our diagram of the previous section:

```python
from discopy.stream import Ty, Stream

N, S = Ty("N"), Ty("S")
A, B = [Stream.sequence(f, N, N) for f in "AB"]
L = Stream.sequence('L', S.head @ N.delay() @ N.delay(), N @ N)
ALB = (L >> A @ B).feedback(dom=S.head, cod=Ty(), mem=N @ N)
ALB.unroll(2).now.foliation().draw()
```

![Alice loves unrolling](https://github.com/discopy/discopy/raw/main/docs/_static/stream/alice-loves-unrolling.png)

Now if we use the [`python`](https://docs.discopy.org/en/main/_api/discopy.python.html) module to interpret each box as a call to a chatbot with the prompt as input, we can get an output along the following lines:

> The play is set in a basement with computers everywhere, Alice and Bob are dressed like hackers with black hoodies and nerdy glasses, they have somewhat of a hipster vibe.
> 
> Alice: I think I’ve cracked the encryption, but it’s like nothing I’ve seen before. DisCoPy — it’s almost...alive.  
> 
> Bob: What do you mean, alive? You’re not saying it’s AI, are you? Because if it is, we’re in way over our heads.
> 
> Alice: It’s not just AI, Bob. It’s adaptive, learning—like it knows we’re here.
> 
> (Bob takes a step back, his face serious as he considers the implications. He glances at the screens around them, suddenly aware of their presence.)
> 
> Bob: If that’s true, we’re not just hacking into the system. We’re waking it up. And if it wakes up angry...
> 
> Alice: Then we’re the ones who let it loose.
> 
> Bob: We need to find the off switch. Now. Before it finds us.
> 
> SILENCE
