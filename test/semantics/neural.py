# -*- coding: utf-8 -*-

from pytest import importorskip, raises

from discopy.neural import *
from discopy.neural import Box, CMap, Diagram, Functor, Id
from discopy.utils import AxiomError


def test_dim():
    assert Dim(2).l == Dim(2).r == Dim(2)
    assert Dim(2, 3).r == Dim(3, 2)
    assert Dim(1) @ Dim(2) == Dim(2) and Dim(2) @ Dim(3) == Dim(2, 3)


def test_axioms():
    x = Dim(2)
    Diagram.use_hypergraph_equality = True
    assert Id(x).transpose() == Id(x) == Id(x).transpose(left=True)
    assert Cap(x, x.r) >> Swap(x, x.r) == Cap(x.r, x)
    assert Swap(x, x.r) >> Cup(x.r, x) == Cup(x, x.r)
    Diagram.use_hypergraph_equality = False


def test_network_as_box():
    f = Network('f', Dim(2), Dim(3), hidden_dim=4, hidden_depth=2)
    g = Network('g', Dim(3), Dim(2))
    assert (f >> g).dom == Dim(2) and (f @ g).cod == Dim(3, 2)
    assert f.dagger().dom == Dim(3) and f.rotate().cod == Dim(2)
    assert f.width == 5 and repr(f) \
        == "neural.Network('f', dom=Dim(2), cod=Dim(3))"
    assert Network('f', Dim(2), Dim(3)) == Network('f', Dim(2), Dim(3))
    one, other = (
        Network('f', Dim(2), Dim(3), data=object()) for _ in range(2))
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


def test_default_module():
    torch = importorskip("torch")
    f = Network('f', Dim(2), Dim(3))
    assert f.module(torch.ones(5, 5)).shape == (5, 5)
    assert f.module is f.data is f.dagger().module
    assert f(torch.ones(1, 5)).shape == (1, 5)
    deep = Network('g', Dim(2), Dim(3), hidden_dim=7, hidden_depth=2).module
    assert deep[0].out_features == 7 and len(deep) == 5


def test_weight_sharing():
    importorskip("torch")
    grid = sudoku(2, 4)
    cell = grid.boxes[0]
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
    f = Network('f', Dim(2), Dim(3))
    x = torch.rand(4, 2)
    expected = f.module(torch.cat([x, torch.zeros(4, 3)], dim=-1))[:, 2:]
    assert torch.allclose(f.to_map()(x), expected)
    assert f.to_map()().shape == (1, 3)


def test_forward_closed_map():
    torch = importorskip("torch")
    torch.manual_seed(0)
    grid = sudoku(2, 3)
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


def test_forward_type_error():
    importorskip("torch")
    box_map = Box('f', Dim(2), Dim(2)).to_map()
    with raises(TypeError):
        box_map.module_list
    with raises(TypeError):
        box_map()


def test_torch_protocol():
    torch = importorskip("torch")
    grid = sudoku(2, 2)
    assert grid.train() is grid and grid.eval() is grid
    assert grid.to(torch.float32) is grid
    assert dict(grid.named_parameters()).keys() \
        == grid.state_dict().keys()
    grid.load_state_dict(grid.state_dict())
    module = grid.as_module()
    assert list(module.parameters()) == list(grid.parameters())
    outer = torch.nn.Sequential(module)
    assert list(outer.parameters()) == list(grid.parameters())
    states = module()
    assert len(states) == 16


def test_training():
    torch = importorskip("torch")
    torch.manual_seed(0)
    n_cells, n_digits, dim = 16, 4, 4
    grid = sudoku(2, dim)
    n_peers = len(sudoku_peers()[0])
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
