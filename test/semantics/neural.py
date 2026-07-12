# -*- coding: utf-8 -*-

from pytest import importorskip, raises

from discopy.neural import *
from discopy.neural import Box, CMap, Diagram, Functor, Id
from discopy.utils import AxiomError


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
    Diagram.use_hypergraph_equality = True
    assert Id(x).transpose() == Id(x) == Id(x).transpose(left=True)
    assert Cap(x, x.r) >> Swap(x, x.r) == Cap(x.r, x)
    assert Swap(x, x.r) >> Cup(x.r, x) == Cup(x, x.r)
    Diagram.use_hypergraph_equality = False


def test_network_as_box():
    f = Network('f', Dim(2), Dim(3))
    g = Network('g', Dim(3), Dim(2))
    assert (f >> g).dom == Dim(2) and (f @ g).cod == Dim(3, 2)
    assert f.dagger().dom == Dim(3) and f.rotate().cod == Dim(2)
    assert repr(f) == "neural.Network('f', dom=Dim(2), cod=Dim(3))"
    assert Network('f', Dim(2), Dim(3)) == Network('f', Dim(2), Dim(3))
    one, other = (
        Network('f', Dim(2), Dim(3), module=object()) for _ in range(2))
    assert one != other and one == one.dagger().dagger()


def test_box_ports():
    f = Network('f', Dim(2, 3), Dim(4, 5, 6))
    fm = f.to_map()
    assert fm.box_ports(0) == (2, 3, 6, 5, 4)
    assert fm.port_widths == (2, 3, 2, 3, 6, 5, 4, 4, 5, 6)


def test_from_wiring():
    f = Network('f', Dim(2), Dim(2, 3))
    g = Network('g', Dim(3), Dim(2, 2))
    cmap = CMap.from_wiring((f, g), [
        ((0, 0), (1, 1)), ((0, 1), (1, 2)), ((0, 2), (1, 0))])
    assert cmap.edges.is_fixpoint_free_involution()
    with raises(ValueError):  # unwired ports
        CMap.from_wiring((f, g), [((0, 0), (1, 1))])
    with raises(ValueError):  # port wired twice
        CMap.from_wiring((f, g), [
            ((0, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 0))])
    with raises(ValueError):  # port wired to itself
        CMap.from_wiring((f, ), [((0, 0), (0, 0))])
    with raises(ValueError):  # no such port
        CMap.from_wiring((f, ), [((0, 0), (0, 3))])
    with raises(AxiomError):  # width mismatch
        CMap.from_wiring((f, g), [
            ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2))])


def test_sudoku_peers():
    peers = sudoku_peers()
    assert set(map(len, peers)) == {7} and peers[0] == (1, 2, 3, 4, 5, 8, 12)
    assert all(cell in peers[peer]
               for cell in range(16) for peer in peers[cell])
    assert set(map(len, sudoku_peers(3))) == {20}


def test_sudoku_map():
    grid = sudoku()
    assert len(grid.boxes) == 16 and grid.n_ports == 112
    assert grid.boxes[0] is grid.boxes[15]
    assert grid.edges.is_fixpoint_free_involution()
    assert len(grid.connected_components) == 1
    assert not grid.is_planar


def test_solve_sudoku():
    solution = solve_sudoku((0, ) * 16)
    assert check_sudoku(solution)
    assert solve_sudoku(solution) == solution
    assert solve_sudoku((1, 1) + (0, ) * 14) is None
    assert solve_sudoku((0, 0, 1, 2, 3, 4) + (0, ) * 10) is None
    assert check_sudoku(solve_sudoku((0, ) * 12 + (1, 2, 3, 0)))
    assert not check_sudoku((0, ) * 16) and not check_sudoku((1, ) * 16)


def test_random_sudoku():
    clues, solution = random_sudoku(seed=42)
    assert check_sudoku(solution)
    assert (clues, solution) == random_sudoku(seed=42)
    assert solution != random_sudoku(seed=43)[1]
    assert sum(clue != 0 for clue in clues) == 8
    assert all(clue in (0, digit) for clue, digit in zip(clues, solution))


def test_decode_sudoku():
    logits = 16 * [[.1, .2, .3, .4]]
    assert decode_sudoku(logits, (0, ) * 16) == (4, ) * 16
    assert decode_sudoku(logits, (1, ) * 16) == (1, ) * 16


def test_network_module():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(3), module=mlp(5))
    assert f.module(torch.ones(4, 5)).shape == (4, 5)
    assert f(torch.ones(1, 5)).shape == (1, 5)
    assert f.module is f.data is f.dagger().module


def test_weight_sharing():
    importorskip("torch")
    cell = Network('cell', Dim(0), Dim(4) ** 7, module=mlp(28))
    grid = sudoku(2, network=cell)
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
    cell = Network('cell', Dim(0), Dim(3) ** 7, module=mlp(21))
    grid = sudoku(2, network=cell)
    states = grid()
    assert len(states) == 16 and all(s.shape == (1, 21) for s in states)
    init = torch.rand(5, sum(grid.port_widths))
    injected, not_injected = (
        grid(init=init, n_rounds=2, inject=inject)
        for inject in (True, False))
    assert not any(map(torch.equal, injected, not_injected))
    loss = sum(state.sum() for state in grid(init=init))
    loss.backward()
    assert all(p.grad is not None for p in grid.parameters())


def test_forward_errors():
    importorskip("torch")
    box_map = Box('f', Dim(2), Dim(2)).to_map()
    with raises(TypeError):
        box_map.module_list
    with raises(TypeError):
        box_map()
    with raises(ValueError):  # a network with no module
        Network('f', Dim(2), Dim(2)).to_map().module_list


def test_torch_protocol():
    torch = importorskip("torch")
    cell = Network('cell', Dim(0), Dim(2) ** 7, module=mlp(14))
    grid = sudoku(2, network=cell)
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
    n_cells, n_digits, dim = 16, 4, 4
    n_peers = len(sudoku_peers()[0])
    cell = Network('cell', Dim(0), Dim(dim) ** n_peers,
                   module=mlp(n_peers * dim))
    grid = sudoku(2, network=cell)
    embedding = torch.nn.Embedding(n_digits + 1, dim)
    readout = torch.nn.Linear(n_peers * dim, n_digits)
    optimizer = torch.optim.Adam([
        *grid.parameters(), *embedding.parameters(),
        *readout.parameters()], lr=0.02)

    clues, solution = random_sudoku(seed=1)
    clues_tensor = torch.tensor([clues])
    target = torch.tensor([solution]) - 1

    def logits():
        embedded = embedding(clues_tensor)
        init = [None] * grid.n_ports
        for box_index in range(n_cells):
            for port in grid.box_ports(box_index):
                init[port] = embedded[:, box_index, :]
        states = grid(init=init, n_rounds=4)
        return readout(torch.stack(states, dim=1))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            logits().reshape(-1, n_digits), target.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    decoded = decode_sudoku(logits()[0], clues)
    assert len(decoded) == n_cells


def test_sudoku_block():
    assert sudoku_block(2) == (2, 2) and sudoku_block(3) == (3, 3)
    assert sudoku_block((2, 3)) == (2, 3) == sudoku_block([2, 3])


def test_rectangular_sudoku_peers():
    peers = sudoku_peers((2, 3))
    assert len(peers) == 36 and set(map(len, peers)) == {12}
    assert all(cell in peers[peer]
               for cell in range(36) for peer in peers[cell])
    assert peers[0] == (1, 2, 3, 4, 5, 6, 7, 8, 12, 18, 24, 30)
    assert sudoku_peers((3, 2)) != peers


def test_rectangular_sudoku_map():
    grid = sudoku((2, 3))
    assert len(grid.boxes) == 36 and grid.n_ports == 432
    assert grid.boxes[0] is grid.boxes[35]
    assert grid.edges.is_fixpoint_free_involution()
    assert len(grid.connected_components) == 1


def test_solve_rectangular_sudoku():
    solution = solve_sudoku(36 * (0, ), (2, 3))
    assert check_sudoku(solution, (2, 3))
    assert solution[:6] == (1, 2, 3, 4, 5, 6)
    assert solve_sudoku(solution, (2, 3)) == solution
    assert solve_sudoku((1, 1) + 34 * (0, ), (2, 3)) is None
    assert not check_sudoku(solution, (3, 2))  # the blocks differ


def test_random_rectangular_sudoku():
    clues, solution = random_sudoku((2, 3), n_clues=12, seed=42)
    assert check_sudoku(solution, (2, 3))
    assert sum(clue != 0 for clue in clues) == 12
    assert all(clue in (0, digit) for clue, digit in zip(clues, solution))
    assert solve_sudoku(clues, (2, 3)) is not None


def test_rectangular_training():
    torch = importorskip("torch")
    torch.manual_seed(0)
    n, n_cells, n_digits, dim = (2, 3), 36, 6, 4
    n_peers = len(sudoku_peers(n)[0])
    cell = Network('cell', Dim(0), Dim(dim) ** n_peers,
                   module=mlp(n_peers * dim))
    grid = sudoku(n, network=cell)
    embedding = torch.nn.Embedding(n_digits + 1, dim)
    readout = torch.nn.Linear(n_peers * dim, n_digits)
    optimizer = torch.optim.Adam([
        *grid.parameters(), *embedding.parameters(),
        *readout.parameters()], lr=0.02)

    clues, solution = random_sudoku(n, n_clues=18, seed=1)
    clues_tensor = torch.tensor([clues])
    target = torch.tensor([solution]) - 1

    def logits():
        embedded = embedding(clues_tensor)
        init = [None] * grid.n_ports
        for box_index in range(n_cells):
            for port in grid.box_ports(box_index):
                init[port] = embedded[:, box_index, :]
        states = grid(init=init, n_rounds=4)
        return readout(torch.stack(states, dim=1))

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            logits().reshape(-1, n_digits), target.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert len(decode_sudoku(logits()[0], clues, n)) == n_cells
