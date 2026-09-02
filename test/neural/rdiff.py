# -*- coding: utf-8 -*-

import subprocess
import sys

from pytest import importorskip, raises

from discopy.neural import Diagram, Dim, Hypergraph, Network
from discopy.neural.rdiff import (
    ReverseRule, differentiate, discard, pair, rdiff, reverse_rule)
from discopy.utils import AxiomError


def make_rule(name, dom, cod, residual):
    return reverse_rule(
        Network(f"{name}.forward", dom, residual @ cod),
        Network(f"{name}.backward", residual @ cod, dom), residual)


def structural_discard(typ):
    return Network("Discard", typ, Dim())


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
        reverse_rule(Network("forward", x, memory @ y),
                     Network("backward", memory @ y, y), memory)
    with raises(AxiomError):
        reverse_rule(Network("forward", x, memory @ y),
                     Network("backward", y @ memory, x), memory)


def test_reverse_rule_composition():
    x, y, z = Dim(2), Dim(3), Dim(5)
    first = make_rule("f", x, y, Dim(7))
    second = make_rule("g", y, z, Dim(11))
    result = first >> second
    assert result.dom == pair(x) and result.cod == pair(z)
    assert result.residual == Dim(7, 11)
    assert result.forward.dom == x and result.forward.cod == Dim(7, 11) @ z
    assert result.backward.dom == Dim(7, 11) @ z and result.backward.cod == x
    assert result.forward == first.forward >> Dim(7) @ second.forward
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
    assert result.forward.cod == Dim(11, 13) @ b @ d
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


def test_rdiff_type_and_discard():
    x, y, memory = Dim(2), Dim(3), Dim(5)
    f = Network("f", x, y)
    result = rdiff(
        f.to_hypergraph(), {f: make_rule("f", x, y, memory)},
        discard_factory=structural_discard)
    assert result.dom == x @ y and result.cod == x
    assert result.is_causal and result.is_monogamous
    assert any(box.name == "Discard" for box in result.boxes)
    with raises(ValueError, match="discard factory"):
        rdiff(
            f.to_hypergraph(), {f: make_rule("f", x, y, memory)},
            discard_factory=lambda typ: Network("bad", Dim(), typ))


def test_default_discard_is_zero():
    torch = importorskip("torch")
    dropped = discard(Dim(2, 3))
    value = torch.randn(4, 5)
    assert torch.equal(dropped.module(value), torch.zeros_like(value))


def test_missing_and_invalid_rules():
    x, y = Dim(2), Dim(3)
    f = Network("f", x, x)
    with raises(ValueError, match="Missing reverse rule"):
        differentiate(f.to_hypergraph(), {})
    with raises(TypeError):
        differentiate(f.to_hypergraph(), {f: object()})
    with raises(ValueError, match="Expected a rule"):
        differentiate(f.to_hypergraph(), {f: make_rule("g", x, y, Dim(3))})
    non_causal = f.to_hypergraph().trace()
    assert non_causal.is_monogamous and not non_causal.is_causal
    with raises(ValueError, match="causality"):
        differentiate(non_causal, {f: make_rule("f", x, x, Dim(3))})
    non_monogamous = Hypergraph.spiders(1, 2, x)
    with raises(ValueError, match="monogamy"):
        differentiate(non_monogamous, {})
