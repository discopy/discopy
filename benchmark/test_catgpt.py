# -*- coding: utf-8 -*-

"""
Conformance tests for the clean-room CatGPT neural diagram.

The small synthetic shape runs without data downloads.  ``CATGPT_FULL=1``
adds a construction-only check of the original six-block shape; a 10,000-step
training run is intentionally not part of the ordinary test suite.
"""

from dataclasses import replace
import os
import time

import pytest

torch = pytest.importorskip("torch")
catgpt = pytest.importorskip("benchmark.catgpt")
CATGPT, CatGPT, CatGPTConfig = (
    catgpt.CATGPT, catgpt.CatGPT, catgpt.CatGPTConfig)
initialise, reference = catgpt.initialise, catgpt.reference


def test_original_shape_and_parameter_count():
    """The published CatGPT shape has eight bias-free matrices."""
    assert (
        CATGPT.vocab,
        CATGPT.batch,
        CATGPT.context,
        CATGPT.width,
        CATGPT.heads,
        CATGPT.blocks,
    ) == (65, 64, 32, 384, 6, 6)
    assert len(CATGPT.parameter_shapes) == 8
    assert CATGPT.parameter_count == 2_704_128
    assert catgpt.attention_scale(CATGPT.width) == 19.595918655395508


def test_forward_conformance():
    """A one-block DisCoPy diagram agrees with direct PyTorch."""
    config = CatGPTConfig.tiny()
    tokens, parameters = initialise(config)
    model = CatGPT(config)

    actual = model(tokens, parameters)
    expected = reference(tokens, parameters, config)

    assert model.graph.is_causal and model.graph.is_monogamous
    assert all(not tuple(module.parameters()) for module in model.cmap.modules)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("softmax", [False, True])
def test_reverse_derivative_and_sgd_step(softmax):
    """The structural VJP and one SGD step agree with autograd."""
    config = CatGPTConfig.tiny()
    tokens, parameters = initialise(config, seed=1)
    cotangent = torch.randn_like(tokens)
    model = CatGPT(config, softmax=softmax)

    token_grad, parameter_gradients = model.vjp(
        tokens, parameters, cotangent)

    reference_tokens = tokens.detach().requires_grad_()
    reference_parameters = {
        name: value.detach().requires_grad_()
        for name, value in parameters.items()}
    output = reference(
        reference_tokens, reference_parameters, config, softmax=softmax)
    output.backward(cotangent)

    # The pinned forward evaluates a float32 approximation of Euler's number
    # while its explicit optic uses the mathematical softmax Jacobian.
    tolerance = 1e-6
    assert torch.allclose(
        token_grad, reference_tokens.grad,
        atol=tolerance, rtol=tolerance)
    gradients = {}
    for name, parameter in reference_parameters.items():
        gradients[name] = parameter_gradients[name]
        assert torch.allclose(
            gradients[name], parameter.grad,
            atol=tolerance, rtol=tolerance)

    rate = 0.1
    actual_parameters = {
        name: value - rate * gradients[name]
        for name, value in parameters.items()}
    expected_parameters = {
        name: value - rate * reference_parameters[name].grad
        for name, value in parameters.items()}
    for name in parameters:
        assert torch.allclose(
            actual_parameters[name], expected_parameters[name],
            atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        model(tokens, actual_parameters),
        reference(tokens, expected_parameters, config, softmax=softmax),
        atol=tolerance, rtol=tolerance)


@pytest.mark.benchmark(
    group="CatGPT structural VJP",
    timer=time.process_time,
    disable_gc=True)
@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_structural_vjp_benchmark(benchmark, n):
    """Record a bounded structural reverse-derivative timing."""
    config = replace(CatGPTConfig.tiny(), blocks=n)
    tokens, parameters = initialise(config)
    model = CatGPT(config, softmax=True)
    cotangent = torch.randn_like(tokens)

    token_gradient, parameter_gradients = benchmark.pedantic(
        lambda: model.vjp(tokens, parameters, cotangent),
        rounds=3, warmup_rounds=1)
    assert token_gradient.shape == tokens.shape
    assert parameter_gradients.keys() == parameters.keys()


def test_tiny_training_smoke():
    """The self-contained structural-VJP training entry point runs."""
    parameters, losses = catgpt.train(
        CatGPTConfig.tiny(), steps=2, dtype=torch.float64)
    assert len(parameters) == 3
    assert len(losses) == 2
    assert torch.isfinite(torch.tensor(losses)).all()


def test_random_batch_uses_true_shifted_windows():
    """Synthetic minibatches use fresh, non-wrapping next-token targets."""
    config = CatGPTConfig.tiny()
    stream = torch.arange(100).remainder(config.vocab)
    generator = torch.Generator().manual_seed(0)
    first = catgpt.random_batch(stream, config, generator)
    second = catgpt.random_batch(stream, config, generator)

    for tokens, targets in (first, second):
        inputs = tokens.argmax(dim=-1)
        assert torch.equal(
            targets[:, :-1], (inputs[:, :-1] + 1) % config.vocab)
        assert torch.equal(
            targets[:, -1], (inputs[:, -1] + 1) % config.vocab)
    assert not torch.equal(first[0], second[0])


@pytest.mark.skipif(
    os.environ.get("CATGPT_FULL") != "1",
    reason="set CATGPT_FULL=1 for one full-shape forward/VJP")
def test_full_six_block_execution():
    """Execute one full forward/VJP without downloading data or weights."""
    with torch.no_grad():
        tokens, parameters = initialise(CATGPT, dtype=torch.float32)
        model = CatGPT()
        output = model(tokens, parameters)
        token_gradient, parameter_gradients = model.vjp(
            tokens, parameters, torch.ones_like(output))

    assert len(model.graph.boxes) == 5 * CATGPT.blocks + 3
    assert output.shape == tokens.shape
    assert token_gradient.shape == tokens.shape
    assert len(parameter_gradients) == 8
    assert torch.isfinite(output).all()
    assert all(
        torch.isfinite(value).all()
        for value in parameter_gradients.values())


@pytest.mark.skipif(
    os.environ.get("CATGPT_10K") != "1",
    reason="set CATGPT_10K=1 for the 10,000-step tiny SGD workload")
def test_full_ten_thousand_step_workload():
    """Run the original step count on the tiny deterministic workload."""
    parameters, losses = catgpt.train(CatGPTConfig.tiny())
    assert len(parameters) == 3 and len(losses) == 10_000
    assert torch.isfinite(torch.tensor(losses)).all()
