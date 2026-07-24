# -*- coding: utf-8 -*-

"""
Conformance tests for the clean-room CatGPT neural diagram.

The small synthetic shape runs without data downloads.  ``CATGPT_FULL=1``
adds a construction-only check of the original six-block shape; a 10,000-step
training run is intentionally not part of the ordinary test suite.
"""

import os

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

    assert torch.allclose(
        token_grad, reference_tokens.grad, atol=1e-9, rtol=1e-9)
    gradients = {}
    for name, parameter in reference_parameters.items():
        gradients[name] = parameter_gradients[name]
        assert torch.allclose(
            gradients[name], parameter.grad, atol=1e-9, rtol=1e-9)

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
            atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        model(tokens, actual_parameters),
        reference(tokens, expected_parameters, config, softmax=softmax),
        atol=1e-10, rtol=1e-10)


def test_tiny_training_smoke():
    """The self-contained structural-VJP training entry point runs."""
    parameters, losses = catgpt.train(
        CatGPTConfig.tiny(), steps=2, dtype=torch.float64)
    assert len(parameters) == 3
    assert losses[1] < losses[0]


@pytest.mark.skipif(
    os.environ.get("CATGPT_FULL") != "1",
    reason="set CATGPT_FULL=1 for the full construction workload")
def test_full_six_block_construction():
    """
    Construct the full graph without downloading data or running training.
    """
    model = CatGPT()
    assert model.config.parameter_count == 2_704_128
    assert len(model.graph.boxes) == 5 * CATGPT.blocks + 3


@pytest.mark.skipif(
    os.environ.get("CATGPT_10K") != "1",
    reason="set CATGPT_10K=1 for the full structural-VJP SGD workload")
def test_full_ten_thousand_step_workload():
    """Run the original step count on deterministic synthetic tokens."""
    parameters, losses = catgpt.train()
    assert len(parameters) == 8 and len(losses) == 10_000
    assert torch.isfinite(torch.tensor(losses)).all()
