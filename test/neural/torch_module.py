# -*- coding: utf-8 -*-

from pytest import importorskip, raises

from discopy.neural.network import Box, Dims, Network
from discopy.utils import AxiomError


def test_module():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    x, h = Dims(4), Dims(8)
    layer = Box('layer', x, h, module=torch.nn.Linear(4, 8))
    relu = Box('relu', h, h, module=torch.nn.ReLU())
    head = Box('head', h, x, module=torch.nn.Linear(8, 4))
    add = Box('add', x @ x, x, module=torch.add)
    residual = Network.copy(x) >> (layer >> relu >> head) @ x >> add
    module = Module(residual)
    inputs = torch.ones(5, 4)
    assert module(inputs).shape == (5, 4) and module.network is residual
    assert list(module.boxes) == [layer.module, relu.module, head.module]
    assert len(list(module.parameters())) == 4
    module(inputs).sum().backward()
    assert all(p.grad is not None for p in module.parameters())
    twice = Module(layer >> head >> layer >> head)
    assert len(list(twice.boxes)) == 2 == len(list(twice.parameters())) // 2
    outputs = Module(layer >> Box('split', h, x @ x, module=lambda v: (
        v[:, :4], v[:, 4:])))(inputs)
    assert tuple(output.shape for output in outputs) == ((5, 4), (5, 4))
    with raises(AxiomError, match="layer: Expected a shape ending in"):
        module(torch.ones(5, 3))
    with raises(TypeError):
        module([1.] * 4)


def test_compile():
    torch = importorskip("torch")
    from discopy.neural.torch import Module
    x = Dims(3)
    layer = Box('layer', x, x, module=torch.nn.Linear(3, 3))
    add = Box('add', x @ x, x, module=torch.add)
    module = Module(Network.copy(x) >> layer @ x >> add)
    inputs = torch.rand(2, 3)
    assert torch.allclose(torch.compile(module)(inputs), module(inputs))
