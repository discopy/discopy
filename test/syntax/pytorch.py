import operator
import pytest
from pytest import mark

torch = pytest.importorskip("torch")
import torch.nn as nn

from discopy.markov import Diagram, Hypergraph
from discopy.pytorch import from_torch


# --- Models for Testing ---
class Identity(nn.Module):
    def forward(self, x):
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        return operator.add(x, self.linear(x))

class SimpleMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, query, key, value):
        return self.attention(query, key, value)[0]


# --- Tests ---
def test_from_torch_identity():
    """Test a pure pass-through model."""
    diagram = from_torch(Identity(), as_hypergraph=False)
    assert isinstance(diagram, Diagram)
    assert len(diagram.boxes) == 1

def test_from_torch_attention():
    """Test multiple inputs and shared parameter generation."""
    diagram = from_torch(SimpleMHA(), as_hypergraph=False)
    assert isinstance(diagram, Diagram)
    assert len(diagram.boxes) > 0
    assert diagram.dom.name == "C"

def test_from_torch_hypergraph_direct():
    """Verify hypergraph generation with no simplification works."""
    hypergraph = from_torch(ResidualBlock(), as_hypergraph=True, simplify=False)
    assert isinstance(hypergraph, Hypergraph)
    assert len(hypergraph.boxes) > 0

@mark.skip(reason="Infinite loop in hypergraph.py simplification (causality vs monogamousity). Handing over to Alexis.")
def test_from_torch_hypergraph_simplify_loop():
    """
    Documents the infinite looping problem in simplification. A Residual block with an add and a copy 
    causes simplify() to loop endlessly when bypassing the 2D layout.
    """
    _ = from_torch(ResidualBlock(), as_hypergraph=True)