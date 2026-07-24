# -*- coding: utf-8 -*-

from pytest import importorskip, raises

from discopy import compact
from discopy.neural import *
from discopy.neural import CMap, Diagram, Functor, Id
from discopy.python.finset import Permutation


def mlp(width):
    torch = importorskip("torch")
    return torch.nn.Sequential(
        torch.nn.Linear(width, 2 * width), torch.nn.Tanh(),
        torch.nn.Linear(2 * width, width))


def test_dim():
    assert Dim(0) == Dim() == Dim(0, 0)
    assert Dim(0) @ Dim(2) == Dim(2) and Dim(2) @ Dim(3) == Dim(2, 3)
    assert Dim(2).l == Dim(2).r == Dim(2)
    assert Dim(2, 3).r == Dim(3, 2)
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


def test_port_dims():
    f = Network('f', Dim(2, 3), Dim(4, 5, 6))
    fm = f.to_map()
    assert fm.port_dims == (2, 3, 2, 3, 6, 5, 4, 4, 5, 6)


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
    assert len(grid.module_list) == 1
    assert sum(p.numel() for p in grid.parameters()) \
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


def test_forward_closed_map():
    torch = importorskip("torch")
    torch.manual_seed(0)
    cell = Network('cell', Dim(0), Dim(3) ** 2, module=mlp(6))
    grid = ring(16, cell)
    states = grid()
    assert len(states) == 16 and all(s.shape == (1, 6) for s in states)
    init = torch.rand(5, sum(grid.port_dims))
    injected, not_injected = (
        grid(init=init, n_rounds=2, inject=inject)
        for inject in (True, False))
    assert not any(map(torch.equal, injected, not_injected))
    loss = sum(state.sum() for state in grid(init=init))
    loss.backward()
    assert all(p.grad is not None for p in grid.parameters())


class NotANetwork(compact.Box, Diagram):
    """ A neural box that is not a network, for negative testing. """


def test_forward_errors():
    importorskip("torch")
    box_map = NotANetwork('f', Dim(2), Dim(2)).to_map()
    with raises(TypeError):
        box_map.module_list
    with raises(TypeError):
        box_map()
    with raises(ValueError):  # a network with no module
        Network('f', Dim(2), Dim(2)).to_map().module_list


def test_torch_protocol():
    torch = importorskip("torch")
    cell = Network('cell', Dim(0), Dim(2) ** 2, module=mlp(4))
    grid = ring(16, cell)
    assert grid.train() is grid and grid.eval() is grid
    assert grid.to(torch.float32) is grid
    assert dict(grid.named_parameters()).keys() \
        == grid.state_dict().keys()
    grid.load_state_dict(grid.state_dict())
    network = grid.as_network()
    assert network.dom == network.cod == Dim(0)
    assert list(network.module.parameters()) == list(grid.parameters())
    outer = torch.nn.Sequential(network.module)
    assert list(outer.parameters()) == list(grid.parameters())
    states = network()
    assert len(states) == 16


def test_training():
    torch = importorskip("torch")
    torch.manual_seed(0)
    n_cells, n_classes, dim = 8, 4, 4
    cell = Network('cell', Dim(0), Dim(dim) ** 2, module=mlp(2 * dim))
    grid = ring(n_cells, cell)
    embedding = torch.nn.Embedding(n_classes, dim)
    readout = torch.nn.Linear(2 * dim, n_classes)
    optimizer = torch.optim.Adam([
        *grid.parameters(), *embedding.parameters(),
        *readout.parameters()], lr=0.02)

    clues = torch.arange(n_cells).remainder(n_classes)[None]
    target = clues.flip(-1)  # each cell must learn its mirror's class

    def logits():
        embedded = embedding(clues)
        init = [None] * grid.n_ports
        for box_index in range(n_cells):
            for port in grid._box_port_indices[box_index]:
                init[port] = embedded[:, box_index, :]
        states = grid(init=init, n_rounds=4)
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
