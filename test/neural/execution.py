# -*- coding: utf-8 -*-

from copy import deepcopy
from itertools import product
import pickle
import random

from pytest import importorskip, raises, skip

from discopy import compact
from discopy.neural import (
    BACKENDS, Backend, CMap, Diagram, Dim, Execution, Id, Network,
    get_backend)
from discopy.neural.backend import backend
from discopy.python.finset import Permutation


def mlp(width):
    torch = importorskip("torch")
    return torch.nn.Sequential(
        torch.nn.Linear(width, 2 * width), torch.nn.Tanh(),
        torch.nn.Linear(2 * width, width))


def ring(n_cells, network):
    """
    A closed ring of identical cells, each wired to its neighbours.

    Assumes an empty domain, wiring the second codomain port of each cell
    to the first codomain port of the next, in the clockwise (i.e.
    reversed) codomain order used by combinatorial maps.
    """
    width = len(network.cod)
    pairs = [(cell * width, ((cell + 1) % n_cells) * width + 1)
             for cell in range(n_cells)]
    edges = Permutation.from_transpositions(pairs, n_cells * width)
    return CMap(CMap.ob(), CMap.ob(), n_cells * (network, ), edges)


def test_backend_contract():
    with raises(TypeError):
        Backend()
    assert set(BACKENDS) == {"pytorch", "jax"}
    importorskip("torch")
    from discopy.neural.torch import PyTorch
    pytorch = PyTorch()
    assert get_backend(pytorch) is pytorch
    assert isinstance(get_backend("pytorch"), PyTorch)
    assert get_backend("pytorch") is get_backend()


def test_get_backend():
    torch = importorskip("torch")
    with raises(KeyError, match="Unknown backend"):
        get_backend("numpy")
    assert get_backend(like=torch.zeros(1)) is get_backend("pytorch")
    assert get_backend(like=object()) is get_backend()
    jax = importorskip("jax")
    with backend("jax") as outer:
        assert get_backend() is outer
        assert get_backend(like=torch.zeros(1)) is get_backend("pytorch")
        assert get_backend(like=jax.numpy.zeros(1)) is outer
        with backend("pytorch") as inner:
            assert get_backend() is inner and inner is not outer
        with backend() as same:
            assert same is outer
    assert get_backend() is get_backend("pytorch")


def test_weight_sharing():
    importorskip("torch")
    cell = Network('cell', Dim(0), Dim(4) ** 2, module=mlp(8))
    grid = ring(6, cell)
    model = grid.as_network().module
    assert len(grid.modules) == len(model.networks) == 1
    assert sum(p.numel() for p in model.parameters()) \
        == sum(p.numel() for p in cell.module.parameters())


def test_runtime_modules():
    torch = importorskip("torch")
    from discopy.neural.torch import PyTorch

    class Scale(torch.nn.Module):
        def __init__(self, scalar):
            super().__init__()
            self.scalar = scalar

        def forward(self, value):
            incoming, outgoing = value.chunk(2, dim=-1)
            del outgoing
            return torch.cat(
                (torch.zeros_like(incoming), self.scalar * incoming), dim=-1)

    original, replacement = Scale(2), Scale(3)
    cell = Network('cell', Dim(1), Dim(1), module=original)
    cmap = (cell >> cell).to_map()
    value = torch.tensor([[5.]])

    assert cmap.module_indices == (0, 0) and cmap.modules == (original, )
    assert torch.equal(
        cmap(value, modules=(replacement, ), causal=True), 9 * value)
    assert torch.equal(
        Execution(
            cmap, value, backend=PyTorch(), modules=(replacement, )
        ).forward_causal(), 9 * value)
    assert torch.equal(cmap(value, n_rounds=2), 4 * value)
    assert torch.equal(
        cmap(value, modules=(replacement, ), n_rounds=2), 9 * value)
    assert len(cmap.step_cache) == 1
    with raises(ValueError, match="Expected 1 modules, got 0"):
        Execution(cmap, value, modules=())


def test_forward_rerouting():
    torch = importorskip("torch")
    x = torch.tensor([[.1, .2, .3, .4, .5]])
    assert (Id(Dim(5)).transpose().to_map()(x) == x).all()
    swapped = Diagram.swap(Dim(2), Dim(3)).to_map()(x)
    assert (swapped == torch.tensor([[.3, .4, .5, .1, .2]])).all()
    cup = Diagram.cups(Dim(2), Dim(2)).to_map()
    assert cup(torch.ones(1, 4)).shape == (1, 0)


def test_forward_open_map():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(3), module=mlp(5))
    x = torch.rand(4, 2)
    expected = f.module(torch.cat([x, torch.zeros(4, 3)], dim=-1))[:, 2:]
    assert torch.allclose(f.to_map()(x), expected)
    assert f.to_map()().shape == (1, 3)

    execution = Execution(f.to_map(), x)
    execution.initialize()
    execution.activate()
    execution.route()
    assert torch.allclose(execution.readout(), expected)


def test_forward_causal_schedule():
    torch = importorskip("torch")

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            incoming, _ = value.chunk(2, dim=-1)
            return torch.cat((torch.zeros_like(incoming), incoming), dim=-1)

    first_module, second_module = Identity(), Identity()
    first = Network('first', Dim(1), Dim(1), module=first_module)
    second = Network('second', Dim(1), Dim(1), module=second_module)
    cmap = (first >> second).to_map()
    value = torch.tensor([[42.]])

    assert torch.equal(cmap(value, causal=True), value)
    assert first_module.calls == second_module.calls == 1
    with raises(ValueError, match="cannot be combined"):
        cmap(value, causal=True, n_rounds=2)
    with raises(ValueError, match="acyclic"):
        first.to_map().trace()(causal=True)

    cell = Network(
        'cell', Dim(0), Dim(1, 1), module=torch.nn.Identity())
    with raises(ValueError, match="causal schedule"):
        ring(2, cell)(causal=True)


def test_topological_order():
    importorskip("torch")
    unit = Dim(1)
    a, b = (Network(name, unit, unit, module=object()) for name in "ab")
    split = Network('split', unit, unit @ unit, module=object())
    merge = Network('merge', unit @ unit, unit, module=object())
    diamond = (split >> a @ b >> merge).to_map()
    assert Execution(diamond).topological_order == (0, 1, 2, 3)
    backwards = CMap(unit, unit, (b, a), Permutation([3, 4, 5, 0, 1, 2]))
    assert backwards.is_monogamous and not backwards.is_causal
    assert Execution(backwards).topological_order == (1, 0)
    assert backwards.topological_order().boxes == (a, b)
    rejected = (
        a.to_map().trace(),
        (a @ Diagram.cups(unit, unit)).to_map(),
        (a.transpose(left=True) >> b.transpose(left=True)).to_map(),
        ring(2, Network('cell', Dim(0), unit @ unit, module=object())),
        CMap(Dim(), Dim(), (), Permutation([]), loops=(unit, )))
    for cmap in rejected:
        with raises(ValueError, match="acyclic monogamous"):
            Execution(cmap).topological_order


def test_private_memory():
    torch = importorskip("torch")

    class Accumulator(torch.nn.Module):
        def forward(self, value):
            incoming, outgoing, memory = value.split((1, 1, 1), dim=-1)
            del outgoing
            next_memory = incoming + memory
            return torch.cat(
                (torch.zeros_like(incoming), next_memory, next_memory),
                dim=-1)

    cell = Network(
        'accumulator', Dim(1), Dim(1), module=Accumulator(), mem=Dim(1))
    cmap = cell.to_map()
    x = torch.tensor([[2.]])

    output, memory = cmap(x, n_rounds=3, return_memory=True)
    assert torch.equal(output, torch.tensor([[6.]]))
    assert torch.equal(memory[0], torch.tensor([[6.]]))

    output, memory = cmap(
        x, memory=torch.tensor([[10.]]),
        n_rounds=2, inject=False, return_memory=True)
    assert torch.equal(output, torch.tensor([[14.]]))
    assert torch.equal(memory[0], torch.tensor([[14.]]))

    wrapped = cmap.as_network()
    output, memory = wrapped(
        x, n_rounds=2, return_memory=True)
    assert torch.equal(output, torch.tensor([[4.]]))
    assert torch.equal(memory[0], torch.tensor([[4.]]))

    assert wrapped.mem == Dim(1)
    output, memory = wrapped.to_map()(
        x, n_rounds=2, return_memory=True)
    assert torch.equal(output, torch.tensor([[4.]]))
    assert torch.equal(memory[0], torch.tensor([[4.]]))


def test_private_memory_per_occurrence():
    torch = importorskip("torch")

    class Counter(torch.nn.Module):
        def forward(self, value):
            public, memory = value.split((1, 1), dim=-1)
            return torch.cat((public, memory + 1), dim=-1)

    cell = Network(
        'counter', Dim(0), Dim(1), module=Counter(), mem=Dim(1))
    cmap = CMap(
        CMap.ob(), CMap.ob(), 2 * (cell, ),
        Permutation.from_transpositions([(0, 1)], 2))
    initial = [torch.tensor([[3.]]), torch.tensor([[7.]])]
    _, memory = cmap(
        memory=initial, n_rounds=2, return_memory=True)

    assert cmap.boxes[0].module is cmap.boxes[1].module
    assert torch.equal(memory[0], torch.tensor([[5.]]))
    assert torch.equal(memory[1], torch.tensor([[9.]]))
    assert memory[0] is not memory[1]


def test_private_memory_validation():
    torch = importorskip("torch")
    cell = Network(
        'stateful', Dim(0), Dim(1),
        module=torch.nn.Identity(), mem=Dim(1))
    cmap = cell.to_map()

    with raises(ValueError, match="init must contain"):
        cmap(init=[])
    with raises(ValueError, match="memory must contain"):
        cmap(memory=[])
    with raises(ValueError, match="memory\\[0\\] has shape"):
        cmap(memory=[torch.zeros(1, 2)])
    malformed = Network(
        'malformed', Dim(0), Dim(1),
        module=torch.nn.Linear(2, 1), mem=Dim(1)).to_map()
    with raises(ValueError, match="output of box 0 has shape"):
        malformed(n_rounds=1)

    public = Network(
        'public', Dim(0), Dim(1), module=torch.nn.Identity()).to_map()
    output = public(init=[None, None], n_rounds=1)
    assert torch.equal(output, torch.zeros(1, 1))


def test_batch_size_disagreement():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(3), module=mlp(6), mem=Dim(1)).to_map()
    with raises(ValueError, match=r"init has shape \(3, 10\), expected \(2"):
        f(torch.zeros(2, 2), init=torch.zeros(3, 10))
    with raises(ValueError, match=r"memory has shape \(3, 1\), expected \(2"):
        f(torch.zeros(2, 2), memory=torch.zeros(3, 1))
    with raises(ValueError, match="x has shape"):
        f(torch.zeros(2, 3))
    with raises(ValueError, match="Messages must have shape"):
        f(torch.zeros(2))
    with raises(ValueError, match="negative"):
        f(n_rounds=-1)
    assert f(memory=torch.zeros(4, 1)).shape == (4, 3)
    closed = ring(2, Network('cell', Dim(0), Dim(1, 1), module=mlp(2)))
    with raises(ValueError, match="takes no input"):
        closed(torch.zeros(1, 1))
    assert all(
        state.shape == (5, 2) for state in closed(torch.zeros(5, 0)))


def test_portless_boxes():
    torch = importorskip("torch")

    class Tick(torch.nn.Module):
        def forward(self, value):
            return value + 1

    f = Network('f', Dim(2), Dim(3), module=mlp(5))
    clock = Network('clock', Dim(), Dim(), module=Tick(), mem=Dim(1))
    _, memory = clock.to_map()(n_rounds=3, return_memory=True)
    assert torch.equal(memory[0], torch.tensor([[3.]]))
    _, memory = clock.to_map()(causal=True, return_memory=True)
    assert torch.equal(memory[0], torch.tensor([[1.]]))
    assert (clock @ f).to_map()(torch.zeros(2, 2)).shape == (2, 3)

    inner = ring(2, Network('cell', Dim(0), Dim(2) ** 2, module=mlp(4)))
    wrapped = inner.as_network('inner')
    for batch_size in (1, 3):
        x = torch.rand(batch_size, 2)
        assert torch.allclose((wrapped @ f).to_map()(x), f.to_map()(x))
    assert wrapped.module.box_forward(torch.zeros(3, 0)).shape == (3, 0)


def test_init_on_boundary_ports():
    torch = importorskip("torch")

    class Double(torch.nn.Module):
        def forward(self, value):
            incoming, _ = value.chunk(2, dim=-1)
            return torch.cat(
                (torch.zeros_like(incoming), 2 * incoming), dim=-1)

    cmap = Network('double', Dim(1), Dim(1), module=Double()).to_map()
    x, bias = torch.tensor([[3.]]), torch.tensor([[1000.]])
    on_output, on_input = ([
        bias if port in ports else None for port in range(cmap.n_ports)]
        for ports in (cmap.output_ports, cmap.input_ports))
    assert torch.equal(cmap(x, n_rounds=1), 2 * x)
    assert torch.equal(cmap(x, init=on_output, n_rounds=1), 2 * x + bias)
    assert torch.equal(
        cmap(x, init=on_output, n_rounds=1, inject=False), 2 * x)
    assert torch.equal(cmap(x, init=on_output, n_rounds=0), bias)
    assert torch.equal(cmap(x, init=on_input, n_rounds=1), 2 * x)
    flat = cmap(x, init=on_input, n_rounds=1, return_flat=True)
    assert torch.equal(cmap.read(flat, cmap.input_ports)[:, 0], bias)


def test_return_rounds_and_flat():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(2), module=mlp(4))
    g = Network('g', Dim(2), Dim(3), module=mlp(5))
    cmap = (f >> g).to_map()
    x = torch.rand(3, 2)
    rounds = cmap(x, n_rounds=3, return_rounds=True)
    assert len(rounds) == 3 and all(
        torch.allclose(state, cmap(x, n_rounds=k + 1))
        for k, state in enumerate(rounds))
    flat = cmap(x, n_rounds=3, return_flat=True)
    assert flat.shape == (3, sum(cmap.port_widths))
    assert torch.allclose(
        cmap.read(flat, cmap.output_ports).reshape(3, -1), rounds[-1])
    flats, memories = cmap(
        x, n_rounds=3, return_rounds=True, return_flat=True,
        return_memory=True)
    assert len(flats) == 3 and torch.equal(flats[-1], flat)
    assert all(memory.shape == (3, 0) for memory in memories)

    flat = cmap(x, causal=True, return_flat=True)
    assert torch.allclose(
        cmap.read(flat, cmap.output_ports).reshape(3, -1),
        cmap(x, causal=True))
    with raises(ValueError, match="no rounds to return"):
        cmap(x, causal=True, return_rounds=True)

    closed = ring(4, Network('cell', Dim(0), Dim(2) ** 2, module=mlp(4)))
    states = closed(n_rounds=2, return_rounds=True)
    assert len(states) == 2 and len(states[1]) == 4
    assert all(
        torch.allclose(one, other)
        for one, other in zip(states[1], closed(n_rounds=2)))


def test_compile():
    torch = importorskip("torch")
    f, g = (Network(name, Dim(2), Dim(2), module=mlp(4)) for name in "fg")
    cmap = (f >> g).to_map()
    x = torch.rand(3, 2)
    eager = cmap(x)
    assert cmap.compile() is cmap and cmap.compile_kwargs == {}
    assert "step_cache" not in cmap.__dict__
    try:
        compiled = cmap(x)
    except Exception as error:
        module = type(error).__module__
        if not module.startswith(("torch._dynamo", "torch._inductor")):
            raise
        skip(f"torch.compile cannot run here: {error}")
    assert torch.allclose(compiled, eager) and len(cmap.step_cache) == 1

    fresh = tuple(mlp(4) for _ in range(2))
    expected = (
        Network('f', Dim(2), Dim(2), module=fresh[0])
        >> Network('g', Dim(2), Dim(2), module=fresh[1])).to_map()(x)
    assert torch.allclose(cmap(x, modules=fresh), expected)
    assert torch.allclose(cmap.as_network().module(x), eager)
    assert cmap(x, inject=False).shape == (3, 2) and len(cmap.step_cache) == 1
    cmap(x).sum().backward()
    assert all(p.grad is not None for p in f.module.parameters())
    assert torch.allclose(
        cmap(x, causal=True), (f >> g).to_map()(x, causal=True))


def last_ready_first(cmap):
    """ A topological order of the boxes taking the last ready box first. """
    ports, order = cmap.ports, []
    dependencies = [set() for _ in cmap.boxes]
    for source, target in cmap.box_edges:
        dependencies[int(ports[target].depth - .5)].add(
            int(ports[source].depth + .5))
    while len(order) < len(cmap.boxes):
        order.append(max(
            index for index in range(len(cmap.boxes))
            if index not in order and dependencies[index] <= set(order)))
    return order


def reference(cmap, x=None, init=None, memory=None, n_rounds=None,
              inject=True, causal=False, modules=None):
    """
    The execution formula written per port with one module call per box:
    the flat incoming messages and the memories, as ``return_flat=True``
    and ``return_memory=True`` return them.
    """
    torch = importorskip("torch")
    widths, edges = cmap.port_widths, cmap.edges
    modules = cmap.modules if modules is None else modules
    rows = next((v.shape[0] for v in (x, init, memory) if v is not None), 1)

    def split(flat, widths):
        if flat is None or not widths:
            return [torch.zeros(rows, width).double() for width in widths]
        return list(torch.split(flat, list(widths), dim=-1))

    initial, memories = split(init, widths), split(memory, cmap.memory_widths)
    source = split(None, widths)
    for port, value in zip(cmap.input_ports, split(
            x, [widths[port] for port in cmap.input_ports])):
        source[port] = value
    inject = inject and init is not None
    incoming = [
        initial[port] + source[edges[port]] for port in range(cmap.n_ports)]

    def fire(index):
        ports = cmap.box_ports(index)
        value = torch.cat(
            [incoming[port] for port in ports] + [memories[index]], dim=-1)
        output = modules[cmap.module_indices[index]](value)
        width = sum(widths[port] for port in ports)
        memories[index] = output[:, width:]
        return dict(zip(ports, split(
            output[:, :width], [widths[port] for port in ports])))

    if causal:
        for index in last_ready_first(cmap):
            for port, value in fire(index).items():
                incoming[edges[port]] = value + (
                    initial[edges[port]] if inject else 0)
        return torch.cat(incoming, dim=-1), tuple(memories)
    for _ in range(len(cmap.boxes) if n_rounds is None else n_rounds):
        outgoing = list(source)
        for index in range(len(cmap.boxes)):
            for port, value in fire(index).items():
                outgoing[port] = value
        incoming = [
            outgoing[edges[port]] + (initial[port] if inject else 0)
            for port in range(cmap.n_ports)]
    return torch.cat(incoming, dim=-1), tuple(memories)


def random_closed_map(cells, n_boxes):
    """ A closed map of random cells, their ports paired at random. """
    boxes = [random.choice(cells) for _ in range(n_boxes)]
    ports = [(index, position) for index, box in enumerate(boxes)
             for position in range(len(box.dom) + len(box.cod))]
    random.shuffle(ports)
    return CMap.from_wiring(boxes, list(zip(ports[::2], ports[1::2])))


def random_open_map(cells, n_steps):
    """ A feed-forward map of random cells, composed when the types match. """
    diagram = random.choice(cells)
    for _ in range(n_steps):
        other = random.choice(cells)
        diagram = diagram >> other if diagram.cod == other.dom\
            else diagram @ other
    return diagram.to_map()


def test_oracle():
    torch = importorskip("torch")
    random.seed(0)
    torch.manual_seed(0)
    cells = (
        Network('a', Dim(2), Dim(2), module=mlp(4).double()),
        Network('b', Dim(2), Dim(2, 2, 2), module=mlp(8).double()),
        Network('c', Dim(2, 2), Dim(2, 2), module=mlp(9).double(),
                mem=Dim(1)),
        Network('d', Dim(0), Dim(2, 2), module=mlp(6).double(), mem=Dim(2)),
        Network('e', Dim(2, 2), Dim(2), module=mlp(6).double()))
    rows = 3

    def randn(width):
        return torch.randn(rows, width, dtype=torch.float64)

    def configs(cmap, causal=False):
        x = randn(sum(cmap.port_widths[i] for i in cmap.input_ports))\
            if cmap.has_boundary else None
        for init, memory, inject, n_rounds in product(
                (None, randn(sum(cmap.port_widths))),
                (None, randn(sum(cmap.memory_widths))),
                (True, False), (None, ) if causal else (1, 3)):
            yield dict(x=x, init=init, memory=memory, inject=inject,
                       n_rounds=n_rounds, causal=causal)

    def same(got, expected):
        return torch.allclose(got[0], expected[0], atol=1e-9) and all(
            torch.allclose(one, other, atol=1e-9)
            for one, other in zip(got[1], expected[1]))

    batched = False
    for _ in range(4):
        closed, open_map = (
            random_closed_map(cells[:4], 5), random_open_map(cells, 5))
        batched |= any(
            len(group["boxes"]) > 1 for cmap in (closed, open_map)
            for group in cmap.routing["groups"])
        for cmap, causal in ((closed, False), (open_map, False),
                             (open_map, True)):
            for kwargs in configs(cmap, causal):
                assert same(
                    cmap(return_flat=True, return_memory=True, **kwargs),
                    reference(cmap, **kwargs))
        replacement = tuple(map(deepcopy, open_map.modules))
        for parameter in torch.nn.ModuleList(replacement).parameters():
            torch.nn.init.normal_(parameter)
        kwargs = dict(
            x=next(configs(open_map))["x"], n_rounds=2, modules=replacement)
        assert same(
            open_map(return_flat=True, return_memory=True, **kwargs),
            reference(open_map, **kwargs))
    assert batched


def test_forward_closed_map():
    torch = importorskip("torch")
    torch.manual_seed(0)
    cell = Network('cell', Dim(0), Dim(3) ** 2, module=mlp(6))
    grid = ring(16, cell)
    network = grid.as_network()
    states = network()
    assert len(states) == 16 and all(s.shape == (1, 6) for s in states)
    assert network(n_rounds=0) == 16 * (None, )
    init = torch.rand(5, sum(grid.port_widths))
    injected, not_injected = (
        network(init=init, n_rounds=2, inject=inject)
        for inject in (True, False))
    assert not any(map(torch.equal, injected, not_injected))
    loss = sum(state.sum() for state in network(init=init))
    loss.backward()
    assert all(p.grad is not None for p in network.module.parameters())


class NotANetwork(compact.Box, Diagram):
    """ A neural box that is not a network, for negative testing. """


def test_forward_errors():
    importorskip("torch")
    box_map = NotANetwork('f', Dim(2), Dim(2)).to_map()
    with raises(TypeError):
        box_map.modules
    with raises(TypeError):
        box_map()
    with raises(ValueError):  # a network with no module
        Network('f', Dim(2), Dim(2)).to_map().modules


def test_torch_wrapper():
    torch = importorskip("torch")
    cell = Network('cell', Dim(0), Dim(2) ** 2, module=mlp(4))
    grid = ring(16, cell)
    network = grid.as_network()
    other = grid.as_network()
    model = network.module
    assert network.dom == network.cod == Dim(0)
    assert model is not other.module and type(model) is type(other.module)
    assert model.networks[0] is other.module.networks[0] is cell.module
    assert model.train() is model and model.eval() is model
    assert model.to(torch.float32) is model
    assert dict(model.named_parameters()).keys() == model.state_dict().keys()
    model.load_state_dict(model.state_dict())
    outer = torch.nn.Sequential(model)
    assert list(outer.parameters()) == list(model.parameters())
    states = network()
    assert len(states) == 16


def test_torch_wrapper_binds_backend():
    torch = importorskip("torch")

    from discopy.neural.torch import PyTorch

    class Exploding(PyTorch):
        def activate(self, module, value):
            raise AssertionError("ambient backend must not be used")

    selected = PyTorch()
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0., 0.], [1., 0.]]))
    wrapped = Network(
        'f', Dim(1), Dim(1), module=module
    ).to_map().as_network(backend=selected)
    value = torch.tensor([[3.]])

    assert wrapped.module.backend is selected
    assert torch.equal(wrapped.module(value, backend=Exploding()), value)


def test_nested_torch_wrapper():
    torch = importorskip("torch")

    class Bidirectional(torch.nn.Module):
        def forward(self, value):
            left, right = value.chunk(2, dim=-1)
            return torch.cat((2 * right, 3 * left), dim=-1)

    module = Bidirectional()
    inner = Network('f', Dim(1), Dim(1), module=module).to_map()
    wrapped = inner.as_network()
    value = torch.tensor([[3.]])

    assert torch.equal(wrapped(value), 3 * value)
    assert torch.equal(wrapped.to_map()(value), 3 * value)
    assert torch.equal(
        wrapped.module.box_forward(torch.tensor([[3., 5.]])),
        torch.tensor([[10., 9.]]))


def test_nested_chain():
    torch = importorskip("torch")

    class FeedForward(torch.nn.Module):
        """ A layer ignoring the messages incoming on its codomain. """
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, value):
            incoming, _ = value.chunk(2, dim=-1)
            return torch.tanh(self.linear(
                torch.cat((incoming, torch.zeros_like(incoming)), dim=-1)))

    def chains(factory):
        first, f, g, h, last = (
            Network(name, Dim(2), Dim(2), module=factory())
            for name in ("first", "f", "g", "h", "last"))
        inner = (f >> g >> h).to_map().as_network('inner')
        return ((first >> inner >> last).to_map(),
                (first >> f >> g >> h >> last).to_map())

    x = torch.rand(3, 2)
    nested, flat = chains(FeedForward)
    assert len(nested.boxes) == 3 and len(flat.boxes) == 5
    assert torch.allclose(nested(x, causal=True), flat(x, causal=True))
    assert torch.allclose(nested(x, n_rounds=3), flat(x, n_rounds=5))
    assert not torch.allclose(nested(x, n_rounds=2), flat(x, n_rounds=5))
    nested, flat = chains(lambda: torch.nn.Linear(4, 4))
    assert not torch.allclose(nested(x, causal=True), flat(x, causal=True))


def test_torch_wrapper_copy_and_pickle():
    torch = importorskip("torch")
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0., 0.], [1., 0.]]))
    cmap = Network('f', Dim(1), Dim(1), module=module).to_map()
    wrapped = cmap.as_network().module
    x = torch.tensor([[3.]])

    clone = deepcopy(wrapped)
    assert clone.inside.boxes[0].module is clone.networks[0]
    with torch.no_grad():
        clone.networks[0].weight.zero_()
    assert torch.equal(wrapped(x), x)
    assert torch.equal(clone(x), torch.zeros_like(x))

    restored = pickle.loads(pickle.dumps(wrapped))
    assert "step_cache" not in restored.inside.__dict__
    assert "index_cache" not in restored.inside.__dict__
    assert restored.inside.boxes[0].module is restored.networks[0]
    assert torch.equal(restored(x), x)


def test_training():
    torch = importorskip("torch")
    torch.manual_seed(0)
    n_cells, n_classes, dim = 8, 4, 4
    cell = Network('cell', Dim(0), Dim(dim) ** 2, module=mlp(2 * dim))
    grid = ring(n_cells, cell)
    network = grid.as_network()
    embedding = torch.nn.Embedding(n_classes, dim)
    readout = torch.nn.Linear(2 * dim, n_classes)
    optimizer = torch.optim.Adam([
        *network.module.parameters(), *embedding.parameters(),
        *readout.parameters()], lr=0.02)

    clues = torch.arange(n_cells).remainder(n_classes)[None]
    target = clues.flip(-1)  # each cell must learn its mirror's class

    def logits():
        embedded = embedding(clues)
        init = [None] * grid.n_ports
        for box_index in range(n_cells):
            for port in grid.box_ports(box_index):
                init[port] = embedded[:, box_index, :]
        states = network(init=init, n_rounds=4)
        return readout(torch.stack(states, dim=1))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            logits().reshape(-1, n_classes), target.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert logits().shape == (1, n_cells, n_classes)
