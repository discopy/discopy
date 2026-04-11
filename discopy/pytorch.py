# -*- coding: utf-8 -*-

"""
PyTorch-to-DisCoPy translation utilities for Markov diagrams.

This module builds Markov representations from torch.fx graphs, using
explicit parameter morphisms (C -> P) and data morphisms (C -> T).

Example
-------
>>> import torch.nn as nn
>>> from discopy.pytorch import from_torch
>>> class SimpleMHA(nn.Module):
>>>     def __init__(self):
>>>         super().__init__()
>>>         self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

>>>     def forward(self, query, key, value):
>>>         return self.attention(query, key, value)[0]

>>> model = SimpleLinearModel()
>>> diagram = from_torch(model)
>>> diagram.draw()
"""

from collections.abc import Callable
from discopy.markov import Box, Ty, Diagram, Hypergraph, Id, Copy, Swap


# --- Object Types ---
C = Ty("C")  # Context: Represents the abstract source
T = Ty("T")  # Tensor: Represents a data wire
P = Ty("P")  # Parameter: Represents a trainable parameter

# --- Registration ---
TRANSLATION_REGISTRY: dict[str, Callable] = {}

def register_translation(type_name: str):
    """
    Decorator to map torch.fx ops to DisCoPy boxes.
    
    Note: 'type_name' must match the torch.fx node.op (e.g., 'placeholder')
    or the name of the module/function being called (e.g., 'Linear').
    """
    def decorator(cls):
        TRANSLATION_REGISTRY[type_name] = cls
        return cls
    return decorator


# --- Translated Boxes ---
# Note: Classes are registered using the exact strings torch.fx uses to identify operations.

@register_translation("get_attr")
class InitParam(Box):
    def __init__(self, name: str):
        super().__init__(name, dom=C, cod=P)

@register_translation("placeholder")
class Placeholder(Box):
    def __init__(self, name: str):
        super().__init__(name, dom=C, cod=T)

@register_translation("Linear")
class Linear(Box):
    def __init__(self, name: str):
        super().__init__(name, dom=T @ P, cod=T)

@register_translation("add")
class Add(Box):
    def __init__(self, name="add"):
        super().__init__(name, dom=T @ T, cod=T)

@register_translation("MultiheadAttention")
class Attention(Box):
    def __init__(self, name: str):
        super().__init__(name, dom=T @ T @ T @ P, cod=T @ T)

@register_translation("getitem")
class Projection(Box):
    def __init__(self, name: str):
        super().__init__(f"proj_{name}", dom=T @ T, cod=T)
