import pytest

torch = pytest.importorskip("torch")

from discopy.grammar.abstract import Ty
from discopy.grammar.neural import *
from discopy.grammar.proofnet import ProofNet

n, s = Ty("n"), Ty("s")
Alice, Bob = n("Alice"), n("Bob")
loves = (n >> (n >> s))("loves")
signature = Signature.from_types([n >> (n >> s)])


def test_Signature():
    x, y, z = map(Ty, "xyz")
    assert Signature.from_types([x >> (y >> x), z]).atoms == (x, y, z)
    for ty in [x, x >> y, (x >> y) >> (z >> x)]:
        assert Signature((x, y, z)).decode(
            Signature((x, y, z)).encode(ty)) == ty
    assert signature.arities == [2, 0, 0] and len(signature) == 3


def test_Tagger_beam():
    torch.manual_seed(0)
    tagger = Tagger(signature, hidden=8, dim=8)
    states = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]) > 0
    beams = tagger.beam(states, mask, k=3, max_length=5)
    assert len(beams) == 5
    for beam in beams:
        scores = [score for _, score in beam]
        assert len(beam) == 3 and scores == sorted(scores, reverse=True)
        assert all(len(signature.encode(ty)) <= 5 for ty, _ in beam)
    assert all(ty in signature.atoms
               for beam in tagger.beam(states, mask, k=2, max_length=1)
               for ty, _ in beam)


def test_sinkhorn():
    blocks = [torch.randn(1, 1, requires_grad=True),
              torch.randn(3, 3, requires_grad=True)]
    log_p = sinkhorn(blocks, iterations=50)
    assert log_p.shape == (2, 3, 3) and log_p[0, 0, 0] == 0
    assert torch.equal(log_p[0].exp(), torch.eye(3))
    log_p[1].diagonal().sum().backward()
    assert all(torch.isfinite(block.grad).all() for block in blocks)


def test_Linker():
    torch.manual_seed(0)
    linker = Linker(signature, hidden=8, dim=8, max_index=2)
    net = ProofNet.from_term(loves(Bob)(Alice), (Alice, loves, Bob))
    scores = linker(torch.randn(3, 8), net.sequent)
    assert scores.shape == (6, 6)
    states, mask = torch.randn(2, 4, 8), torch.tensor([[1, 1, 1, 0]] * 2) > 0
    loss = linker.loss(states, mask, [net, net])
    loss.backward()
    assert loss > 0 and torch.isfinite(linker.bilinear.grad).all()
