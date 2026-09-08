# -*- coding: utf-8 -*-

import subprocess
import sys

from pytest import importorskip, raises

from discopy.neural import Diagram, Dim, Hypergraph, Network, Permutation
from discopy.neural.rdiff import (
    ReverseRule, differentiate, discard, generator_rule, pair, rdiff,
    reverse_rule)
from discopy.utils import AxiomError


def make_rule(name, dom, cod, residual):
    return reverse_rule(
        Network(f"{name}.forward", dom, cod @ residual),
        Network(f"{name}.backward", residual @ cod, dom), residual)


def structural_discard(typ):
    return Network("Discard", typ, Dim())


def autograd_rules(torch):
    """
    Reverse rules whose legs are all-port modules: the forward leg emits
    ``f(x)`` then its input as the residual, the backward leg the
    vector-Jacobian product ``vjp(x, dy)`` of the residual and cotangent.
    """
    class Forward(torch.nn.Module):
        def __init__(self, a, f):
            super().__init__()
            self.a, self.f = a, f

        def forward(self, value):
            x = value[:, :self.a]
            return torch.cat((torch.zeros_like(x), self.f(x), x), dim=-1)

    class Backward(torch.nn.Module):
        def __init__(self, a, b, vjp):
            super().__init__()
            self.a, self.b, self.vjp = a, b, vjp

        def forward(self, value):
            x, dy = value[:, :self.a], value[:, self.a:self.a + self.b]
            zeros = torch.zeros_like(value[:, :self.a + self.b])
            return torch.cat((zeros, self.vjp(x, dy)), dim=-1)

    def rule(name, a, b, f, vjp):
        box = Network(name, Dim(a), Dim(b))
        return box, reverse_rule(
            Network(f"{name}.fwd", Dim(a), Dim(b, a), module=Forward(a, f)),
            Network(f"{name}.bwd", Dim(a, b), Dim(a),
                    module=Backward(a, b, vjp)),
            Dim(a))

    return rule


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.neural.rdiff; "
        "assert 'torch' not in sys.modules"], check=True)


def test_reverse_rule_validation():
    x, y, memory = Dim(2), Dim(3), Dim(5)
    rule = make_rule("f", x, y, memory)
    assert isinstance(rule, ReverseRule)
    assert (rule.dom, rule.cod, rule.residual) == (pair(x), pair(y), memory)
    with raises(AxiomError):
        reverse_rule(Network("forward", x, y @ memory),
                     Network("backward", memory @ y, y), memory)
    with raises(AxiomError):
        reverse_rule(Network("forward", x, y @ memory),
                     Network("backward", y @ memory, x), memory)


def test_reverse_rule_composition():
    x, y, z = Dim(2), Dim(3), Dim(5)
    first = make_rule("f", x, y, Dim(7))
    second = make_rule("g", y, z, Dim(11))
    result = first >> second
    assert result.dom == pair(x) and result.cod == pair(z)
    assert result.residual == Dim(7, 11)
    assert result.forward.dom == x and result.forward.cod == z @ Dim(7, 11)
    assert result.backward.dom == Dim(7, 11) @ z and result.backward.cod == x
    assert result.forward == first.forward >> second.forward @ Dim(7)\
        >> z @ Diagram.swap(Dim(11), Dim(7))
    assert result.backward == Dim(7) @ second.backward >> first.backward
    with raises(AxiomError):
        first >> make_rule("h", z, x, Dim(13))


def test_reverse_rule_tensor():
    a, b, c, d = Dim(2), Dim(3), Dim(5), Dim(7)
    left = make_rule("f", a, b, Dim(11))
    right = make_rule("g", c, d, Dim(13))
    result = left @ right
    assert result.dom == pair(a @ c) and result.cod == pair(b @ d)
    assert result.residual == Dim(11, 13)
    assert result.forward.cod == b @ d @ Dim(11, 13)
    assert result.backward.dom == Dim(11, 13) @ b @ d


def test_identity_and_swap_are_structural():
    x, y = Dim(2), Dim(3)
    identity = differentiate(Hypergraph.id(x), {})
    assert identity == ReverseRule.id(pair(x))
    swapped = differentiate(Hypergraph.swap(x, y), {})
    assert swapped.dom == pair(x @ y) and swapped.cod == pair(y @ x)
    assert swapped.residual == Dim()
    assert swapped.forward == Diagram.swap(x, y)
    assert swapped.backward == Diagram.swap(y, x)


def test_permutation_rule():
    perm = Permutation(Dim(2, 3, 5), [1, 2, 0])
    rule = generator_rule(perm, {})
    assert rule.residual == Dim()
    assert rule.forward.to_hypergraph() == perm.to_hypergraph()
    assert rule.backward.to_hypergraph() == perm.dagger().to_hypergraph()


def test_differentiate_composition_and_tensor():
    x, y, z, w = Dim(2), Dim(3), Dim(5), Dim(7)
    f, g = Network("f", x, y), Network("g", y, z)
    h = Network("h", x, w)
    rules = {
        f: make_rule("f", x, y, Dim(11)),
        g: make_rule("g", y, z, Dim(13)),
        h: make_rule("h", x, w, Dim(17)),
    }
    composite = differentiate((f >> g).to_hypergraph(), rules)
    assert composite.cod == pair(z)
    assert composite.residual == Dim(11, 13)
    parallel = differentiate((f @ h).to_hypergraph(), rules)
    assert parallel.cod == pair(y @ w)
    assert parallel.residual == Dim(11, 17)
    out_of_order = (f >> g).to_hypergraph().interchange(0, 1)
    assert out_of_order.is_acyclic and not out_of_order.is_causal
    assert differentiate(out_of_order, rules) == composite
    assert differentiate((f >> g).to_hypergraph(), lambda box: rules[box])\
        == composite


def test_rdiff_type_and_discard():
    x, y, memory = Dim(2), Dim(3), Dim(5)
    f = Network("f", x, y)
    rule = make_rule("f", x, y, memory)
    result = rdiff(
        f.to_hypergraph(), {f: rule}, discard_factory=structural_discard)
    assert isinstance(result, Diagram)
    assert result.dom == x @ y and result.cod == x
    assert result == rule.forward @ y\
        >> structural_discard(y) @ memory @ y >> rule.backward
    graph = result.to_hypergraph()
    assert graph.is_causal and graph.is_monogamous
    with raises(AxiomError):
        rdiff(f.to_hypergraph(), {f: rule},
              discard_factory=lambda typ: Network("bad", Dim(), typ))


def test_rdiff_to_the_unit():
    x, memory = Dim(3), Dim(5)
    sink = Network("sink", x, Dim())
    rule = make_rule("sink", x, Dim(), memory)
    assert discard(Dim()) == Diagram.id(Dim())
    assert rdiff(sink.to_hypergraph(), {sink: rule})\
        == rule.forward >> rule.backward


def test_default_discard_is_zero():
    torch = importorskip("torch")
    dropped = discard(Dim(2, 3))
    value = torch.randn(4, 5)
    assert torch.equal(dropped.module(value), torch.zeros_like(value))


def test_rdiff_against_autograd():
    torch = importorskip("torch")
    torch.manual_seed(0)
    rule, double = autograd_rules(torch), dict(dtype=torch.float64)
    W, V = torch.randn(4, 3, **double), torch.randn(2, 4, **double)
    lin, lin_rule = rule("lin", 3, 4, lambda x: x @ W.T, lambda x, dy: dy @ W)
    act, act_rule = rule(
        "tanh", 4, 4, torch.tanh, lambda x, dy: (1 - torch.tanh(x) ** 2) * dy)
    out, out_rule = rule("out", 4, 2, lambda x: x @ V.T, lambda x, dy: dy @ V)
    derivative = rdiff((lin >> act >> out).to_hypergraph(),
                       {lin: lin_rule, act: act_rule, out: out_rule})
    x, dy = torch.randn(5, 3, **double), torch.randn(5, 2, **double)
    actual = derivative.to_map()(torch.cat((x, dy), dim=-1), causal=True)
    x.requires_grad_()
    (torch.tanh(x @ W.T) @ V.T).backward(dy)
    assert torch.allclose(actual, x.grad)


def test_rdiff_swap_between_boxes():
    torch = importorskip("torch")
    torch.manual_seed(0)
    rule, double = autograd_rules(torch), dict(dtype=torch.float64)
    linear = lambda name, W: rule(
        name, W.shape[1], W.shape[0], lambda x: x @ W.T, lambda x, dy: dy @ W)
    Wf, Wg, Wh, Wk = (torch.randn(*shape, **double)
                      for shape in [(3, 2), (2, 2), (1, 2), (1, 3)])
    (f, rf), (g, rg), (h, rh), (k, rk) = (
        linear(name, W) for name, W in zip("fghk", (Wf, Wg, Wh, Wk)))
    diagram = f @ g >> Diagram.swap(Dim(3), Dim(2)) >> h @ k
    derivative = rdiff(diagram.to_hypergraph(), {f: rf, g: rg, h: rh, k: rk})
    x, dy = torch.randn(5, 4, **double), torch.randn(5, 2, **double)
    actual = derivative.to_map()(torch.cat((x, dy), dim=-1), causal=True)
    x.requires_grad_()
    torch.cat((x[:, 2:] @ Wg.T @ Wh.T, x[:, :2] @ Wf.T @ Wk.T), dim=-1)\
        .backward(dy)
    assert torch.allclose(actual, x.grad)


def test_rdiff_to_the_unit_runs():
    torch = importorskip("torch")
    torch.manual_seed(0)
    rule = autograd_rules(torch)
    sink, sink_rule = rule(
        "sink", 3, 0, lambda x: x[:, :0], lambda x, dy: 2 * x)
    derivative = rdiff(sink.to_hypergraph(), {sink: sink_rule})
    value = torch.randn(4, 3)
    assert torch.allclose(derivative.to_map()(value, causal=True), 2 * value)


def test_missing_and_invalid_rules():
    x, y = Dim(2), Dim(3)
    f = Network("f", x, x)
    rules = {f: make_rule("f", x, x, Dim(3))}
    with raises(ValueError, match="Missing reverse rule"):
        differentiate(f.to_hypergraph(), {})
    with raises(ValueError, match="Missing reverse rule"):
        differentiate(f[::-1].to_hypergraph(), rules)
    with raises(TypeError):
        differentiate(f.to_hypergraph(), {f: object()})
    with raises(ValueError, match="Expected a rule"):
        differentiate(f.to_hypergraph(), {f: make_rule("g", x, y, Dim(3))})
    cyclic = f.to_hypergraph().trace()
    assert cyclic.is_monogamous and not cyclic.is_acyclic
    with raises(ValueError, match="acyclic"):
        differentiate(cyclic, rules)
    non_monogamous = Hypergraph.spiders(1, 2, x)
    with raises(ValueError, match="monogamy"):
        differentiate(non_monogamous, {})
