# -*- coding: utf-8 -*-

from pytest import importorskip

from discopy.neural import (
    CMap, Dim, Id, JAX, Network, backend, get_backend)
from discopy.neural_rdiff import discard
from discopy.python.finset import Permutation

jax = importorskip("jax")
jnp = importorskip("jax.numpy")


def bidirectional(weight, value):
    """ Scale messages travelling in both directions. """
    left, right = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((weight * right, weight * left), axis=-1)


def accumulator(weight, value):
    """ Add the weighted input to private memory and emit the result. """
    incoming, outgoing, memory = jnp.split(value, (1, 2), axis=-1)
    del outgoing
    next_memory = memory + weight * incoming
    return jnp.concatenate(
        (jnp.zeros_like(incoming), next_memory, next_memory), axis=-1)


def module(function=bidirectional, weight=2.):
    """ Make a callable JAX PyTree with one array parameter. """
    return jax.tree_util.Partial(function, jnp.asarray(weight))


def ring(n_cells, network):
    """ Wire two ports on each cell to its neighbours in a closed ring. """
    pairs = [
        (2 * cell, 2 * ((cell + 1) % n_cells) + 1)
        for cell in range(n_cells)]
    edges = Permutation.from_transpositions(pairs, 2 * n_cells)
    return CMap(CMap.ob(), CMap.ob(), n_cells * (network, ), edges)


def test_jax_backend_eager_and_closed():
    selected = get_backend("jax")
    assert isinstance(selected, JAX)

    value = jnp.array([[1., 2.]])
    snake = Id(Dim(2)).transpose().to_map()
    assert jnp.array_equal(snake(value, backend=selected), value)

    open_map = Network(
        "open", Dim(1), Dim(1), module=module()).to_map()
    assert jnp.array_equal(
        open_map(jnp.array([[3.]]), backend=selected), jnp.array([[6.]]))

    cell = Network(
        "cell", Dim(0), Dim(1, 1), module=module())
    model = ring(2, cell).as_network(backend=selected).module
    states = model(n_rounds=1)
    assert len(states) == 2
    assert all(state.shape == (1, 2) for state in states)

    with backend("jax"):
        zero = discard(Dim(2)).module
    assert jnp.array_equal(
        jax.jit(zero)(value), jnp.zeros_like(value))


def test_jax_jit_gradient_update_and_sharing():
    shared = module()
    cell = Network("cell", Dim(1), Dim(1), module=shared)
    cmap = (cell >> cell).to_map()
    model = cmap.as_network(backend="jax").module
    value = jnp.array([[3.]])
    apply = jax.jit(
        lambda current, x: current(x, causal=True))

    assert cmap.module_indices == (0, 0)
    assert len(model.modules) == 1
    assert len(jax.tree_util.tree_leaves(model)) == 1
    assert jnp.array_equal(apply(model, value), 4 * value)

    gradient = jax.grad(
        lambda current: current(value, causal=True).sum())(model)
    assert jnp.array_equal(
        jax.tree_util.tree_leaves(gradient)[0], jnp.array(12.))

    updated = jax.tree.map(lambda parameter: parameter / 2, model)
    assert jnp.array_equal(apply(updated, value), value)
    assert jnp.array_equal(apply(model, value), 4 * value)


def test_nested_jax_wrapper_is_one_pytree():
    cell = Network(
        "cell", Dim(1), Dim(1), module=module())
    inner = cell.to_map().as_network(name="inner", backend="jax")
    outer = inner.to_map().as_network(name="outer", backend="jax").module
    value = jnp.array([[3.]])

    assert len(jax.tree_util.tree_leaves(outer)) == 1
    result = jax.jit(lambda current, x: current(x))(outer, value)
    assert jnp.array_equal(result, 2 * value)
    gradient = jax.grad(lambda current: current(value).sum())(outer)
    assert jnp.array_equal(
        jax.tree_util.tree_leaves(gradient)[0], value.sum())


def test_jax_private_memory_under_jit():
    cell = Network(
        "accumulator", Dim(1), Dim(1),
        module=module(accumulator), mem=Dim(1))
    model = cell.to_map().as_network(backend="jax").module
    value = jnp.array([[2.]])
    apply = jax.jit(lambda current, x: current(
        x, n_rounds=3, return_memory=True))

    output, memories = apply(model, value)
    assert jnp.array_equal(output, jnp.array([[12.]]))
    assert len(memories) == 1
    assert jnp.array_equal(memories[0], jnp.array([[12.]]))
