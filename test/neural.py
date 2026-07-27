# -*- coding: utf-8 -*-

from copy import deepcopy
import pickle
import subprocess
import sys

from pytest import importorskip, raises

from discopy import compact
from discopy.neural import *
from discopy.neural import CMap, Diagram, Functor, Id
from discopy.python.finset import Permutation
from discopy.utils import dumps, loads


def mlp(width):
    torch = importorskip("torch")
    return torch.nn.Sequential(
        torch.nn.Linear(width, 2 * width), torch.nn.Tanh(),
        torch.nn.Linear(2 * width, width))


def test_lazy_torch_import():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import discopy.neural; "
        "assert 'torch' not in sys.modules"], check=True)


def test_dim():
    assert Dim(0) == Dim() == Dim(0, 0)
    assert Dim(0) @ Dim(2) == Dim(2) and Dim(2) @ Dim(3) == Dim(2, 3)
    assert Dim(2).l == Dim(2).r == Dim(2)
    assert Dim(2, 3).r == Dim(3, 2)
    assert loads(dumps(Dim(2, 3))) == Dim(2, 3)
    with raises(ValueError):
        Dim(-1)


def test_axioms():
    x = Dim(2)
    assert Equation(
        Id(x).transpose(), Id(x), Id(x).transpose(left=True))
    assert Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
    assert Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))


def test_network_as_box():
    f = Network('f', Dim(2), Dim(3))
    g = Network('g', Dim(3), Dim(2))
    assert (f >> g).dom == Dim(2) and (f @ g).cod == Dim(3, 2)
    assert f.dagger().dom == Dim(3) and f.rotate().cod == Dim(2)
    assert repr(f) == "neural.Network('f', Dim(2), Dim(3))"
    assert Network('f', Dim(2), Dim(3)) == Network('f', Dim(2), Dim(3))
    one, other = (
        Network('f', Dim(2), Dim(3), module=object()) for _ in range(2))
    assert one != other and one == one.dagger().dagger()
    stateful = Network('f', Dim(2), Dim(3), mem=Dim(4))
    assert stateful != f
    assert stateful.dagger().mem == stateful.rotate().mem == Dim(4)
    assert stateful == stateful.dagger().dagger()
    assert stateful.to_map().port_dims == f.to_map().port_dims
    assert loads(dumps(stateful)) == stateful


def test_port_dims():
    f = Network('f', Dim(2, 3), Dim(4, 5, 6))
    fm = f.to_map()
    assert fm.port_dims == (2, 3, 2, 3, 6, 5, 4, 4, 5, 6)


def test_to_hypergraph():
    f = Network('f', Dim(2), Dim(3), mem=Dim(4))
    hypergraph = f.to_hypergraph()
    round_trip = hypergraph.to_diagram()
    assert round_trip.to_hypergraph() == hypergraph
    assert tuple(round_trip.boxes) == (f, )


def test_network_module():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(3), module=mlp(5))
    assert f.module(torch.ones(4, 5)).shape == (4, 5)
    assert f(torch.ones(1, 5)).shape == (1, 5)
    assert f.module is f.data is f.dagger().module


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


def test_weight_sharing():
    importorskip("torch")
    cell = Network('cell', Dim(0), Dim(4) ** 2, module=mlp(8))
    grid = ring(6, cell)
    model = grid.as_network().module
    assert len(grid.modules) == len(model.networks) == 1
    assert sum(p.numel() for p in model.parameters()) \
        == sum(p.numel() for p in cell.module.parameters())


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


def test_forward_closed_map():
    torch = importorskip("torch")
    torch.manual_seed(0)
    cell = Network('cell', Dim(0), Dim(3) ** 2, module=mlp(6))
    grid = ring(16, cell)
    network = grid.as_network()
    states = network()
    assert len(states) == 16 and all(s.shape == (1, 6) for s in states)
    init = torch.rand(5, sum(grid.port_dims))
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
            for port in grid._box_port_indices[box_index]:
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
