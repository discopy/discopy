# -*- coding: utf-8 -*-

from pytest import importorskip

from discopy.neural import CMap, Diagram, Dim, Id, Network, get_backend
from discopy.neural.backend import backend
from discopy.neural.rdiff import discard
from discopy.python.finset import Permutation

jax = importorskip("jax")
jnp = importorskip("jax.numpy")

from discopy.neural.jax import JAX  # noqa: E402  (needs jax installed)


def bidirectional(weight, value):
    """ Scale messages travelling in both directions. """
    left, right = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((weight * right, weight * left), axis=-1)


def feedforward(weight, value):
    """ Scale the input, ignoring the message incoming on the codomain. """
    incoming, _ = jnp.split(value, 2, axis=-1)
    return jnp.concatenate(
        (jnp.zeros_like(incoming), jnp.tanh(weight * incoming)), axis=-1)


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
    swap = Diagram.swap(Dim(1), Dim(1)).to_map()
    assert jnp.array_equal(
        swap(value, backend=selected), jnp.array([[2., 1.]]))
    cup = Diagram.cups(Dim(1), Dim(1)).to_map()
    assert cup(value, backend=selected).shape == (1, 0)

    open_map = Network(
        "open", Dim(1), Dim(1), module=module()).to_map()
    assert jnp.array_equal(
        open_map(jnp.array([[3.]]), backend=selected), jnp.array([[6.]]))

    cell = Network(
        "cell", Dim(0), Dim(1, 1), module=module())
    model = ring(2, cell).as_network(backend=selected).module
    assert model.backend is selected
    states = model(
        init=jnp.array([[1., 2., 3., 4.]]),
        n_rounds=1, inject=False)
    assert all(map(jnp.array_equal, states, (
        jnp.array([[2., 4.]]), jnp.array([[6., 8.]]))))
    assert model(n_rounds=0) == (None, None)

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

    updated = jax.tree_util.tree_map(
        lambda parameter, tangent: parameter - .01 * tangent,
        model, gradient)
    assert apply(updated, value).sum() < apply(model, value).sum()
    assert jnp.array_equal(apply(model, value), 4 * value)


def test_jax_inject_init_under_jit():
    cell = Network("cell", Dim(1), Dim(1), module=module())
    model = cell.to_map().as_network(backend="jax").module
    x, init = jnp.array([[3.]]), jnp.array([[1., 2., 3., 4.]])
    apply = jax.jit(
        lambda current, x, init, inject: current(
            x, init=init, n_rounds=1, inject=inject),
        static_argnames="inject")
    assert jnp.array_equal(apply(model, x, init, True), jnp.array([[14.]]))
    assert jnp.array_equal(apply(model, x, init, False), jnp.array([[10.]]))
    assert jnp.array_equal(model(x, init=init), jnp.array([[14.]]))
    assert jnp.array_equal(
        model(x, init=init, inject=False), jnp.array([[10.]]))


def test_jax_return_rounds_and_flat():
    cell = Network("cell", Dim(1), Dim(1), module=module())
    model = cell.to_map().as_network(backend="jax").module
    x, init = jnp.array([[3.]]), jnp.array([[1., 2., 3., 4.]])
    rounds = jax.jit(lambda current, x, init: current(
        x, init=init, n_rounds=2, return_rounds=True))(model, x, init)
    assert len(rounds) == 2
    assert all(jnp.array_equal(state, jnp.array([[14.]])) for state in rounds)
    flat = jax.jit(lambda current, x, init: current(
        x, init=init, n_rounds=1, return_flat=True))(model, x, init)
    assert jnp.array_equal(flat, jnp.array([[7., 5., 3., 14.]]))
    flats = model(
        x, init=init, n_rounds=2, return_rounds=True, return_flat=True)
    assert len(flats) == 2 and jnp.array_equal(flats[1], flat)
    assert jnp.array_equal(
        model(x, init=init, causal=True, return_flat=True), flat)


def test_jax_compile():
    cell = Network("cell", Dim(1), Dim(1), module=module())
    cmap = (cell >> cell).to_map()
    x = jnp.array([[3.]])
    eager = cmap(x, backend="jax")
    assert cmap.compile() is cmap
    assert jnp.array_equal(cmap(x, backend="jax"), eager)
    assert jnp.array_equal(cmap(x, backend="jax", inject=False), eager)
    assert len(cmap.step_cache) == 1
    model = cmap.as_network(backend="jax").module

    def loss(current):
        return current(x).sum()

    for _ in range(3):
        gradient = jax.grad(loss)(model)
    assert jnp.array_equal(
        jax.tree_util.tree_leaves(gradient)[0], jnp.array(12.))
    jitted = jax.tree_util.tree_leaves(jax.jit(jax.grad(loss))(model))[0]
    assert jnp.array_equal(jitted, jnp.array(12.))
    assert len(cmap.step_cache) == 1


def test_nested_jax_wrapper_is_one_pytree():
    cell = Network(
        "cell", Dim(1), Dim(1), module=module())
    inner = cell.to_map().as_network(name="inner", backend="jax")
    outer = inner.to_map().as_network(name="outer", backend="jax").module
    value = jnp.array([[3.]])

    assert len(jax.tree_util.tree_leaves(outer)) == 1
    result = jax.jit(lambda current, x: current(x))(outer, value)
    assert jnp.array_equal(result, 2 * value)
    assert jnp.array_equal(
        outer.box_forward(jnp.array([[3., 5.]])),
        jnp.array([[10., 6.]]))
    gradient = jax.grad(lambda current: current(value).sum())(outer)
    assert jnp.array_equal(
        jax.tree_util.tree_leaves(gradient)[0], value.sum())


def test_nested_jax_chain():
    first, f, g, h, last = (
        Network(name, Dim(1), Dim(1), module=module(feedforward, weight))
        for name, weight in zip(
            ("first", "f", "g", "h", "last"), (1., 2., 3., 4., 5.)))
    inner = (f >> g >> h).to_map().as_network("inner", backend="jax")
    nested = (first >> inner >> last).to_map().as_network(
        "nested", backend="jax").module
    flat = (first >> f >> g >> h >> last).to_map().as_network(
        "flat", backend="jax").module
    x = jnp.array([[.5], [-.25], [1.]])
    assert len(jax.tree_util.tree_leaves(nested)) == 5
    assert jnp.allclose(nested(x, causal=True), flat(x, causal=True))
    assert jnp.allclose(nested(x, n_rounds=3), flat(x, n_rounds=5))
    assert not jnp.allclose(nested(x, n_rounds=2), flat(x, n_rounds=5))
    jitted = jax.jit(lambda current, x: current(x, causal=True))(nested, x)
    assert jnp.allclose(jitted, flat(x, causal=True))


def test_jax_private_memory_under_jit():
    cell = Network(
        "accumulator", Dim(1), Dim(1),
        module=module(accumulator, weight=1.), mem=Dim(1))
    model = cell.to_map().as_network(backend="jax").module
    value = jnp.array([[2.]])
    apply = jax.jit(lambda current, x: current(
        x, n_rounds=3, return_memory=True))

    output, memories = apply(model, value)
    assert jnp.array_equal(output, jnp.array([[6.]]))
    assert len(memories) == 1
    assert jnp.array_equal(memories[0], jnp.array([[6.]]))

    output, memories = jax.jit(lambda current, x, memory: current(
        x, memory=memory, n_rounds=2, return_memory=True))(
            model, value, jnp.array([[10.]]))
    assert jnp.array_equal(output, jnp.array([[14.]]))
    assert jnp.array_equal(memories[0], jnp.array([[14.]]))


def test_read_write_across_backends():
    torch = importorskip("torch")
    cmap = Network('f', Dim(2), Dim(3), module=module()).to_map()
    with backend("jax"):
        state = cmap.zeros(2)
    assert isinstance(state, jax.Array)
    values = jnp.ones((2, 1, 2))
    written = cmap.write(state, (0, ), values)
    assert isinstance(written, jax.Array)
    assert jnp.array_equal(cmap.read(written, (0, )), values)
    assert jnp.array_equal(cmap.read(written, (1, )), jnp.zeros((2, 1, 2)))

    torch_state = cmap.zeros(2)
    assert isinstance(torch_state, torch.Tensor)
    with backend("jax"):
        assert isinstance(cmap.zeros(2), jax.Array)
        assert isinstance(cmap.zeros(2, like=torch_state), torch.Tensor)
        written = cmap.write(torch_state, (1, ), torch.ones(2, 1, 2))
        assert isinstance(written, torch.Tensor)
        assert torch.equal(cmap.read(written, (1, )), torch.ones(2, 1, 2))
