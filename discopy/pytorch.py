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


# --- Helper Utilities for Planar Diagrams ---

def _count_uses(arg, target_node):
    """
    Recursively count how many times target_node appears in arg.
    """
    if arg == target_node:
        return 1
    if isinstance(arg, (tuple, list)):
        return sum(_count_uses(a, target_node) for a in arg)
    if isinstance(arg, dict):
        return sum(_count_uses(v, target_node) for v in arg.values())
    return 0

def _get_fan_out(node, graph):
    """
    Determine the number of successor nodes that consume this node's output.
    """
    count = 0
    for n in graph.nodes:
        if n.op == "output": continue
        count += _count_uses(n.args, node) + _count_uses(n.kwargs, node)
    return count

def _simplify_planar(diagram: Diagram) -> Diagram:
    """
    Perform a pass of 'peephole' optimizations, like removing self-cancelling Swaps.
    """
    if not diagram: return diagram
    changed = True
    while changed:
        changed = False
        boxes, offsets = list(diagram.boxes), list(diagram.offsets)
        new_boxes, new_offsets = [], []
        i = 0
        while i < len(boxes):
            if (i + 1 < len(boxes) and isinstance(boxes[i], Swap) and isinstance(boxes[i + 1], Swap)
                and offsets[i] == offsets[i + 1] and boxes[i].dom == boxes[i + 1].cod):
                changed = True
                i += 2
                continue
            new_boxes.append(boxes[i])
            new_offsets.append(offsets[i])
            i += 1
        if changed:
            diagram = Diagram.decode(diagram.dom, list(zip(new_boxes, new_offsets)))
    return diagram

def _collect_nodes(arg):
    """
    Flatten nested arguments to find all torch.fx.Node instances.
    """
    if isinstance(arg, (tuple, list)):
        return sum([_collect_nodes(a) for a in arg], [])
    if isinstance(arg, dict):
        return sum([_collect_nodes(v) for v in arg.values()], [])
    return [arg] if hasattr(arg, "op") else []


# --- Core Builders ---

def _build_hypergraph(model, traced, simplify) -> Hypergraph:
    """
    Convert a traced torch.fx graph into a DisCoPy Hypergraph.
    """
    context_spider, spider_count = 0, 1
    node_to_spiders, generated_params = {}, {}
    boxes, box_wires = [], []
    cod_spiders = []

    for node in traced.graph.nodes:
        if node.op == "output":
            for arg in _collect_nodes(node.args):
                if arg in node_to_spiders:
                    cod_spiders.extend(node_to_spiders[arg])
            continue

        lookup_key = node.op if node.op in ["placeholder", "get_attr"] else \
                     type(model.get_submodule(node.target)).__name__ if node.op == "call_module" else \
                     node.target.__name__

        box = TRANSLATION_REGISTRY[lookup_key](node.name)

        param_spider = None
        if node.op == "call_module":
            if node.target not in generated_params:
                p_box = TRANSLATION_REGISTRY["get_attr"](f"param_{node.target}")
                boxes.append(p_box)
                box_wires.append(((context_spider,), (spider_count,)))
                generated_params[node.target] = spider_count
                spider_count += 1
            param_spider = generated_params[node.target]

        in_spiders = [context_spider] if node.op in ["placeholder", "get_attr"] else []
        if node.op not in ["placeholder", "get_attr"]:
            for arg in _collect_nodes(node.args):
                if arg in node_to_spiders:
                    in_spiders.extend(node_to_spiders[arg])
            if param_spider is not None:
                in_spiders.append(param_spider)

        out_spiders = [spider_count + i for i in range(len(box.cod))]
        spider_count += len(box.cod)
        node_to_spiders[node] = out_spiders

        boxes.append(box)
        box_wires.append((tuple(in_spiders), tuple(out_spiders)))

    cod_ty = Ty().tensor(*[T for _ in cod_spiders]) if cod_spiders else Ty()
    hypergraph = Hypergraph(
        dom=C, 
        cod=cod_ty, 
        boxes=tuple(boxes), 
        wires=((context_spider,), tuple(box_wires), tuple(cod_spiders))
    )
    # .simplify() here causes infinite loop bug with models like the residual block.
    return hypergraph.simplify() if simplify else hypergraph


def _build_diagram(model, traced, simplify) -> Diagram:
    """
    Convert a traced torch.fx graph into a planar 2D Diagram.
    """
    diagram, active_wires = None, []
    generated_params, init_count = {}, 0

    def route_inputs(diag, wires, inputs):
        """Permute existing wires to match the order required by the next box."""
        if not inputs: return diag, wires
        used, selected = set(), []
        for inp in inputs:
            for i, w in enumerate(wires):
                if i not in used and w == inp:
                    selected.append(i); used.add(i)
                    break
        rest = [i for i in range(len(wires)) if i not in used]
        perm = rest + selected
        if perm == list(range(len(wires))): return diag, wires
        return diag.permute(*perm), [wires[i] for i in perm]

    for node in traced.graph.nodes:
        if node.op == "output": continue

        lookup_key = node.op if node.op in ["placeholder", "get_attr"] else \
                     type(model.get_submodule(node.target)).__name__ if node.op == "call_module" else \
                     node.target.__name__
        box = TRANSLATION_REGISTRY[lookup_key](node.name)

        param_name = None
        if node.op == "call_module":
            param_name = f"param_{node.target}"
            if node.target not in generated_params:
                p_box = TRANSLATION_REGISTRY["get_attr"](param_name)
                calls = sum(1 for n in traced.graph.nodes if n.target == node.target and n.op == "call_module")
                if calls > 1: p_box = p_box >> Copy(p_box.cod, calls)
                diagram = p_box if diagram is None else diagram @ p_box
                init_count += 1
                active_wires.extend([param_name] * calls)
                generated_params[node.target] = param_name

        if node.op in ["placeholder", "get_attr"]:
            diagram = box if diagram is None else diagram @ box
            init_count += 1
            active_wires.extend([node] * len(box.cod))
        else:
            inputs = _collect_nodes(node.args) + ([param_name] if param_name else [])
            diagram, active_wires = route_inputs(diagram, active_wires, inputs)
            left_ty = diagram.cod[:-len(box.dom)] if len(box.dom) > 0 else diagram.cod
            diagram = diagram >> (Id(left_ty) @ box if left_ty else box)
            if len(box.dom) > 0: active_wires = active_wires[:-len(box.dom)]
            active_wires.extend([node] * len(box.cod))

        fan_out = _get_fan_out(node, traced.graph)
        if fan_out > 1:
            copy_box = Copy(box.cod, fan_out)
            left_ty = diagram.cod[:-len(box.cod)] if len(box.cod) > 0 else diagram.cod
            diagram = diagram >> (Id(left_ty) @ copy_box if left_ty else copy_box)
            bases = active_wires[-len(box.cod):]
            active_wires = active_wires[:-len(box.cod)]
            for _ in range(fan_out): active_wires.extend(bases)

    if not diagram: return Id()
    if init_count > 1: diagram = Copy(C, init_count) >> diagram

    return _simplify_planar(diagram) if simplify else diagram


# --- Transpiler API ---
def from_torch(model, as_hypergraph: bool = False, simplify: bool = True):
    """
    Translates a PyTorch module into a DisCoPy Markov representation via torch.fx.

    Parameters
    ----------
    model : torch.nn.Module
        Model to trace and translate.
    as_hypergraph : bool
        If True, returns a raw `Hypergraph` (bypassing 2D layout constraints).
        If False, returns a 2D planar `Diagram`.
    """
    try:
        import torch.fx as fx
    except ImportError:
        raise ImportError("PyTorch is required.")

    traced = fx.symbolic_trace(model)
    
    if as_hypergraph:
        return _build_hypergraph(model, traced, simplify)
    else:
        return _build_diagram(model, traced, simplify)

# Inject into Diagram
Diagram.from_torch = staticmethod(from_torch)