# -*- coding: utf-8 -*-

"""
A clean-room CatGPT conformance model built from neural string diagrams.

The original benchmark is a small character transformer with deliberately
few ingredients: one-hot tokens, bias-free projections, pre-normalised
causal attention blocks and no feed-forward sublayers.  This module records
that architecture without copying its implementation, weights or data.
It is a behavior-level reimplementation of the architecture at the
`pinned upstream CatGPT commit
<https://github.com/statusfailed/catgpt/tree/\
c06c7e752b8c5fbca638b60caf79ed6dc878d90e>`_, motivated by
`DisCoPy issue #458 <https://github.com/discopy/discopy/issues/458>`_.
No code or artifacts from the upstream repository are included.

Weights are explicit boundary values rather than hidden module parameters.
For a model input ``X`` and parameter tuple ``P``, the diagram has type
``X @ P -> Y``.  Every primitive carries an explicit local reverse rule, so
:func:`discopy.neural_rdiff.rdiff` builds the model VJP compositionally.

The full six-block shape is useful for checking parameter counts.  Execution
and derivative conformance normally use :meth:`CatGPTConfig.tiny`; set
``CATGPT_FULL=1`` in :mod:`benchmark.test_catgpt` to opt into the full shape.
Set ``CATGPT_10K=1`` to opt into the original 10,000 SGD steps on deterministic
synthetic tokens.  No dataset or pretrained weights are downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import prod, sqrt

import torch

from discopy import neural
from discopy.neural_rdiff import ReverseRule, rdiff


@dataclass(frozen=True)
class CatGPTConfig:
    """Shape of the clean-room CatGPT architecture."""

    vocab: int = 65
    batch: int = 64
    context: int = 32
    width: int = 384
    heads: int = 6
    blocks: int = 6
    eps: float = 1e-5

    def __post_init__(self):
        values = (
            self.vocab, self.batch, self.context,
            self.width, self.heads)
        if any(value <= 0 for value in values):
            raise ValueError("CatGPT dimensions must be positive.")
        if self.blocks < 0:
            raise ValueError("The number of blocks cannot be negative.")
        if self.width % self.heads:
            raise ValueError("The model width must be divisible by heads.")

    @classmethod
    def tiny(cls) -> CatGPTConfig:
        """Return the deterministic shape used by conformance tests."""
        return cls(
            vocab=7, batch=2, context=4,
            width=8, heads=2, blocks=1)

    @property
    def parameter_shapes(self) -> tuple[tuple[str, tuple[int, int]], ...]:
        """Parameter names and shapes in PyTorch ``(output, input)`` order."""
        qkv = tuple(
            (f"blocks.{i}.qkv", (3 * self.width, self.width))
            for i in range(self.blocks))
        return (
            ("token", (self.width, self.vocab)),
            *qkv,
            ("output", (self.vocab, self.width)))

    @property
    def parameter_count(self) -> int:
        """Number of scalar weights."""
        return sum(prod(shape) for _, shape in self.parameter_shapes)


CATGPT = CatGPTConfig()
assert (
    CATGPT.vocab,
    CATGPT.batch,
    CATGPT.context,
    CATGPT.width,
    CATGPT.heads,
    CATGPT.blocks,
    CATGPT.parameter_count,
) == (65, 64, 32, 384, 6, 6, 2_704_128)


def attention_scale(width: int) -> float:
    """Reproduce CatGPT's float32-derived ``sqrt(model_width)``."""
    return torch.sqrt(torch.tensor(width)).item()


class Operation:
    """
    A differentiable primitive with a closed-form vector-Jacobian product.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the primitive."""
        raise NotImplementedError

    def reverse(
            self, value: torch.Tensor,
            cotangent: torch.Tensor) -> torch.Tensor:
        """Pull ``cotangent`` back to the primitive input."""
        raise NotImplementedError


class Linear(Operation):
    """Bias-free projection with its matrix supplied after the input."""

    def __init__(self, items: int, input_width: int, output_width: int):
        self.items = items
        self.input_width = input_width
        self.output_width = output_width
        self.weight_width = input_width * output_width
        self.dom_dims = (
            items * input_width, self.weight_width)
        self.cod_dims = (items * output_width, )

    @property
    def dom_width(self) -> int:
        """Combined flattened input and weight width."""
        return self.items * self.input_width + self.weight_width

    @property
    def cod_width(self) -> int:
        """Flattened output width."""
        return self.items * self.output_width

    def split(self, value):
        """Split and reshape a flattened input and matrix."""
        inputs, weight = value.split((
            self.items * self.input_width, self.weight_width), dim=-1)
        inputs = inputs.reshape(
            -1, self.items, self.input_width)
        weight = weight.reshape(
            -1, self.output_width, self.input_width)
        return inputs, weight

    def forward(self, value):
        inputs, weight = self.split(value)
        return torch.einsum("bti,boi->bto", inputs, weight).flatten(1)

    def reverse(self, value, cotangent):
        inputs, weight = self.split(value)
        cotangent = cotangent.reshape(
            -1, self.items, self.output_width)
        input_grad = torch.einsum(
            "bto,boi->bti", cotangent, weight)
        weight_grad = torch.einsum(
            "bto,bti->boi", cotangent, inputs)
        return torch.cat(
            (input_grad.flatten(1), weight_grad.flatten(1)), dim=-1)


class LayerNorm(Operation):
    """Non-affine layer normalisation with population variance."""

    def __init__(self, items: int, width: int, eps: float = 1e-5):
        self.items, self.width, self.eps = items, width, eps
        self.dom_width = self.cod_width = items * width
        self.dom_dims = self.cod_dims = (items * width, )

    def normalise(self, value):
        """Return normalised values and inverse standard deviation."""
        value = value.reshape(-1, self.items, self.width)
        centred = value - value.mean(dim=-1, keepdim=True)
        inverse_std = (
            centred.square().mean(dim=-1, keepdim=True) + self.eps
        ).rsqrt()
        return centred * inverse_std, inverse_std

    def forward(self, value):
        normalised, _ = self.normalise(value)
        return normalised.flatten(1)

    def reverse(self, value, cotangent):
        normalised, inverse_std = self.normalise(value)
        cotangent = cotangent.reshape(-1, self.items, self.width)
        centred_cotangent = cotangent - cotangent.mean(
            dim=-1, keepdim=True)
        projection = (
            cotangent * normalised).mean(dim=-1, keepdim=True)
        return (
            inverse_std
            * (centred_cotangent - normalised * projection)
        ).flatten(1)


class CausalSelfAttention(Operation):
    """Multi-head causal attention scaled by ``sqrt(model_width)``."""

    def __init__(
            self, batch: int, context: int, width: int, heads: int):
        if width % heads:
            raise ValueError("The attention width must divide into heads.")
        self.batch = batch
        self.context, self.width, self.heads = context, width, heads
        self.head_width = width // heads
        self.scale = attention_scale(width)
        self.dom_width = 3 * batch * context * width
        self.cod_width = batch * context * width
        self.dom_dims, self.cod_dims = (
            (self.dom_width, ), (self.cod_width, ))

    def split(self, value):
        """Split flattened QKV and move the head axis before time."""
        qkv = value.reshape(
            -1, self.batch, self.context, 3, self.width)
        q, k, v = qkv.unbind(dim=3)

        def heads(value):
            return value.reshape(
                -1, self.batch, self.context,
                self.heads, self.head_width
            ).transpose(2, 3)

        return heads(q), heads(k), heads(v)

    def attend(self, value):
        """Return heads, probabilities and the attended output."""
        query, key, content = self.split(value)
        scores = query @ key.transpose(-2, -1) / self.scale
        mask = torch.ones(
            self.context, self.context,
            dtype=torch.bool, device=value.device).triu(1)
        probabilities = scores.masked_fill(mask, -torch.inf).softmax(dim=-1)
        output = probabilities @ content
        return query, key, content, probabilities, output

    def forward(self, value):
        *_, output = self.attend(value)
        return output.transpose(2, 3).reshape(-1, self.cod_width)

    def reverse(self, value, cotangent):
        query, key, content, probabilities, _ = self.attend(value)
        output_grad = cotangent.reshape(
            -1, self.batch, self.context,
            self.heads, self.head_width
        ).transpose(2, 3)
        content_grad = probabilities.transpose(-2, -1) @ output_grad
        probability_grad = output_grad @ content.transpose(-2, -1)
        score_grad = probabilities * (
            probability_grad
            - (probability_grad * probabilities).sum(
                dim=-1, keepdim=True))
        query_grad = score_grad @ key / self.scale
        key_grad = score_grad.transpose(-2, -1) @ query / self.scale

        def merge(value):
            return value.transpose(2, 3).reshape(
                -1, self.batch, self.context, self.width)

        return torch.stack(tuple(map(
            merge, (query_grad, key_grad, content_grad))), dim=3
        ).flatten(1)


class Copy(Operation):
    """Copy a value for a residual branch."""

    def __init__(self, width: int):
        self.dom_width, self.cod_width = width, 2 * width
        self.dom_dims, self.cod_dims = (width, ), (width, width)

    def forward(self, value):
        return torch.cat((value, value), dim=-1)

    def reverse(self, value, cotangent):
        del value
        left, right = cotangent.chunk(2, dim=-1)
        return left + right


class Add(Operation):
    """Add the two sides of a residual branch."""

    def __init__(self, width: int):
        self.dom_width, self.cod_width = 2 * width, width
        self.dom_dims, self.cod_dims = (width, width), (width, )

    def forward(self, value):
        left, right = value.chunk(2, dim=-1)
        return left + right

    def reverse(self, value, cotangent):
        del value
        return torch.cat((cotangent, cotangent), dim=-1)


class Softmax(Operation):
    """Token-wise softmax, optional at the model output."""

    def __init__(self, items: int, width: int):
        self.items, self.width = items, width
        self.dom_width = self.cod_width = items * width
        self.dom_dims = self.cod_dims = (items * width, )

    def forward(self, value):
        return value.reshape(
            -1, self.items, self.width).softmax(dim=-1).flatten(1)

    def reverse(self, value, cotangent):
        output = self.forward(value).reshape(
            -1, self.items, self.width)
        cotangent = cotangent.reshape(-1, self.items, self.width)
        return (
            output * (
                cotangent
                - (cotangent * output).sum(dim=-1, keepdim=True))
        ).flatten(1)


class Arrow(torch.nn.Module):
    """Adapt an operation to a bidirectional neural generator."""

    def __init__(
            self, operation: Operation, dom_width: int, cod_width: int,
            keep_input: bool = False, reverse: bool = False):
        super().__init__()
        self.operation = operation
        self.dom_width, self.cod_width = dom_width, cod_width
        self.keep_input, self.is_reverse = keep_input, reverse

    def forward(self, messages):
        """Read domain messages and emit only on codomain ports."""
        value = messages[:, :self.dom_width]
        if self.is_reverse:
            residual_width = self.operation.dom_width
            residual, cotangent = value.split((
                residual_width, self.operation.cod_width), dim=-1)
            output = self.operation.reverse(residual, cotangent)
        else:
            output = self.operation.forward(value)
            if self.keep_input:
                output = torch.cat((output, value), dim=-1)
        if output.shape[-1] != self.cod_width:
            raise ValueError(
                f"Operation returned width {output.shape[-1]}, "
                f"expected {self.cod_width}.")
        return torch.cat((torch.zeros_like(value), output), dim=-1)


@dataclass
class Primitive:
    """A neural generator paired with its explicit local reverse rule."""

    box: neural.Network
    rule: ReverseRule


def primitive(name: str, operation: Operation) -> Primitive:
    """Turn an operation and its closed-form VJP into neural generators."""
    dom, cod = (
        neural.Dim(*operation.dom_dims), neural.Dim(*operation.cod_dims))
    box = neural.Network(
        name, dom, cod,
        module=Arrow(operation, operation.dom_width, operation.cod_width))
    forward = neural.Network(
        f"{name}.forward", dom, cod @ dom,
        module=Arrow(
            operation, operation.dom_width,
            operation.cod_width + operation.dom_width, keep_input=True))
    reverse = neural.Network(
        f"{name}.reverse", dom @ cod, dom,
        module=Arrow(
            operation, operation.dom_width + operation.cod_width,
            operation.dom_width, reverse=True))
    return Primitive(box, ReverseRule(forward, reverse, cod=cod))


class CatGPT:
    """A CatGPT-shaped neural hypergraph and its local reverse rules."""

    def __init__(self, config: CatGPTConfig = CATGPT, softmax: bool = False):
        self.config, self.softmax = config, softmax
        self.rules = {}
        self.diagram = self._build()
        self.graph = self.diagram.to_hypergraph()
        if not self.graph.is_causal or not self.graph.is_monogamous:
            raise ValueError("The CatGPT graph must be causal and monogamous.")

    def add(self, name, operation):
        """Register a primitive and return its generator."""
        result = primitive(name, operation)
        self.rules[result.box] = result.rule
        return result.box

    def _build(self):
        config = self.config
        context, width = config.context, config.width
        items = config.batch * context
        shapes = config.parameter_shapes
        parameters = tuple(neural.Dim(prod(shape)) for _, shape in shapes)

        token = self.add(
            "Token", Linear(items, config.vocab, width))
        diagram = token @ neural.Id(neural.Dim().tensor(*parameters[1:]))
        state = neural.Dim(items * width)

        for i in range(config.blocks):
            remaining = neural.Dim().tensor(*parameters[i + 2:])
            copy = self.add(f"Copy[{i}]", Copy(items * width))
            norm = self.add(
                f"LayerNorm[{i}]",
                LayerNorm(items, width, config.eps))
            qkv = self.add(
                f"QKV[{i}]", Linear(items, width, 3 * width))
            attention = self.add(
                f"Attention[{i}]",
                CausalSelfAttention(
                    config.batch, context, width, config.heads))
            add = self.add(f"Add[{i}]", Add(items * width))
            diagram >>= copy @ neural.Id(
                parameters[i + 1] @ remaining)
            diagram >>= neural.Id(state) @ norm @ neural.Id(
                parameters[i + 1] @ remaining)
            diagram >>= neural.Id(state) @ qkv @ neural.Id(remaining)
            diagram >>= neural.Id(state) @ attention @ neural.Id(remaining)
            diagram >>= add @ neural.Id(remaining)

        output_norm = self.add(
            "LayerNorm[output]",
            LayerNorm(items, width, config.eps))
        output = self.add(
            "Output", Linear(items, width, config.vocab))
        diagram >>= output_norm @ neural.Id(parameters[-1])
        diagram >>= output
        if self.softmax:
            diagram >>= self.add(
                "Softmax", Softmax(items, config.vocab))
        return diagram

    @cached_property
    def cmap(self):
        """The primal graph as an executable combinatorial map."""
        return self.graph.to_map()

    @cached_property
    def reverse(self):
        """The structurally composed reverse derivative hypergraph."""
        return rdiff(self.graph, self.rules)

    @cached_property
    def reverse_map(self):
        """The reverse derivative as an executable combinatorial map."""
        return self.reverse.to_map()

    def pack(self, tokens, parameters):
        """Pack tokens and shared matrices into the diagram boundary."""
        config = self.config
        expected = (config.batch, config.context, config.vocab)
        if tuple(tokens.shape) != expected:
            raise ValueError(
                f"tokens has shape {tuple(tokens.shape)}, "
                f"expected {expected}.")
        values = [tokens.flatten().unsqueeze(0)]
        for name, shape in self.config.parameter_shapes:
            parameter = parameters[name]
            if tuple(parameter.shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(parameter.shape)}, "
                    f"expected {shape}.")
            values.append(parameter.flatten().unsqueeze(0))
        return torch.cat(values, dim=-1)

    def unpack_gradient(self, value):
        """Split a diagram VJP into input and shared-parameter gradients."""
        config = self.config
        widths = (
            config.batch * config.context * config.vocab,
            *(prod(shape) for _, shape in config.parameter_shapes))
        values = value.split(widths, dim=-1)
        token_grad = values[0].reshape(
            config.batch, config.context, config.vocab)
        parameter_grads = {
            name: item.reshape(shape)
            for (name, shape), item
            in zip(config.parameter_shapes, values[1:])}
        return token_grad, parameter_grads

    def __call__(self, tokens, parameters):
        output = self.cmap(self.pack(tokens, parameters))
        width = self.config.vocab
        return output.reshape(
            self.config.batch, self.config.context, width)

    def vjp(self, tokens, parameters, cotangent):
        """Evaluate the DisCoPy-composed reverse derivative."""
        boundary = torch.cat(
            (self.pack(tokens, parameters),
             cotangent.flatten().unsqueeze(0)), dim=-1)
        return self.unpack_gradient(self.reverse_map(boundary))


def initialise(
        config: CatGPTConfig, seed: int = 0,
        dtype: torch.dtype = torch.float64):
    """Return deterministic synthetic one-hot tokens and matrices."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        config.vocab, (config.batch, config.context), generator=generator)
    tokens = torch.nn.functional.one_hot(
        indices, config.vocab).to(dtype=dtype)
    parameters = {
        name: torch.randn(
            shape, generator=generator, dtype=dtype) / sqrt(shape[1])
        for name, shape in config.parameter_shapes}
    return tokens, parameters


def reference(
        tokens: torch.Tensor, parameters: dict[str, torch.Tensor],
        config: CatGPTConfig, softmax: bool = False) -> torch.Tensor:
    """Independent direct PyTorch expression for conformance comparisons."""
    state = torch.nn.functional.linear(tokens, parameters["token"])
    mask = torch.ones(
        config.context, config.context,
        dtype=torch.bool, device=tokens.device).triu(1)
    for i in range(config.blocks):
        normalised = torch.nn.functional.layer_norm(
            state, (config.width,), eps=config.eps)
        qkv = torch.nn.functional.linear(
            normalised, parameters[f"blocks.{i}.qkv"])
        query, key, content = qkv.chunk(3, dim=-1)
        head_width = config.width // config.heads

        def heads(value):
            return value.reshape(
                -1, config.context, config.heads, head_width
            ).transpose(1, 2)

        query, key, content = map(heads, (query, key, content))
        scores = (
            query @ key.transpose(-2, -1)
            / attention_scale(config.width))
        probability = scores.masked_fill(
            mask, -torch.inf).softmax(dim=-1)
        attended = (probability @ content).transpose(1, 2).reshape_as(state)
        state = state + attended
    state = torch.nn.functional.layer_norm(
        state, (config.width,), eps=config.eps)
    result = torch.nn.functional.linear(state, parameters["output"])
    return result.softmax(dim=-1) if softmax else result


def negative_log_likelihood_vjp(probability, targets):
    """Return mean negative log likelihood and its probability cotangent."""
    if targets.shape != probability.shape[:-1]:
        raise ValueError(
            f"targets has shape {tuple(targets.shape)}, "
            f"expected {tuple(probability.shape[:-1])}.")
    index = targets.unsqueeze(-1)
    target_probability = probability.gather(dim=-1, index=index)
    loss = -target_probability.log().mean()
    cotangent = torch.zeros_like(probability)
    cotangent.scatter_(
        -1, index, -target_probability.reciprocal() / targets.numel())
    return loss, cotangent


def train(
        config: CatGPTConfig = CATGPT, steps: int = 10_000,
        rate: float = .1, seed: int = 0,
        dtype: torch.dtype = torch.float32):
    """
    Run structural-VJP SGD on a deterministic synthetic next-token batch.

    This mirrors the original benchmark's 10,000-step, learning-rate ``.1``
    workload while remaining self-contained.  The full call is intentionally
    opt-in because reverse evaluation is interpreted directly as a DisCoPy
    message-passing graph rather than compiled native code.
    """
    if steps < 0:
        raise ValueError("The number of steps cannot be negative.")
    tokens, parameters = initialise(config, seed=seed, dtype=dtype)
    targets = tokens.argmax(dim=-1).roll(-1, dims=-1)
    model, losses = CatGPT(config, softmax=True), []
    for _ in range(steps):
        probability = model(tokens, parameters)
        loss, cotangent = negative_log_likelihood_vjp(
            probability, targets)
        _, gradients = model.vjp(tokens, parameters, cotangent)
        parameters = {
            name: value - rate * gradients[name]
            for name, value in parameters.items()}
        losses.append(loss.item())
    return parameters, losses
