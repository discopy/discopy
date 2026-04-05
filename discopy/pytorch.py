from collections.abc import Callable
from discopy.markov import Ty, Box


# --- Object Types ---
C = Ty("C")  # Context: Represents the abstract "source"
T = Ty("T")  # Tensor: Represents a data wire (e.g., an activation)
P = Ty("P")  # Parameter: Represents a trainable parameter

# --- Registration ---
TRANSLATION_REGISTRY: dict[str, Callable] = {}


def register_translation(type_name: str):
    """
    Maps the exact string name of a PyTorch class, FX node op, or Python operator to a Box.
    
    Rules for 'type_name':
    - PyTorch Layers: Use type(module).__name__ (e.g., "Linear", "MultiheadAttention")
    - FX Graph Nodes: Use node.op (e.g., "placeholder", "get_attr")
    - Python Operators: Use func.__name__ (e.g., "add", "getitem")
    """
    def decorator(cls):
        TRANSLATION_REGISTRY[type_name] = cls
        return cls
    return decorator


# --- Translated Boxes ---
@register_translation("get_attr")
class InitParam(Box):
    """
    Lifts the abstract context C into a Parameter P.
    """
    def __init__(self, name: str):
        super().__init__(name, dom=C, cod=P)


@register_translation("placeholder")
class Placeholder(Box):
    """
    Represents the entry point of data into the model.
    """
    def __init__(self, name: str):
        super().__init__(name, dom=C, cod=T)


@register_translation("Linear")
class Linear(Box):
    def __init__(self, name: str):
        super().__init__(name, dom=T @ P, cod=T)


@register_translation("add")
class Add(Box):
    """
    Merges two tensor paths via addition.
    """
    def __init__(self, name="add"):
        super().__init__(name, dom=T @ T, cod=T)


@register_translation("MultiheadAttention")
class Attention(Box):
    """
    Standard Multi-Head Attention operation.
    """
    def __init__(self, name: str):
        super().__init__(name, dom=T @ T @ T, cod=T @ T)


@register_translation("getitem")
class Projection(Box):
    """
    Represents selecting a specific output from a product.
    """
    def __init__(self, name: str):
        super().__init__(f"proj_{name}", dom=T @ T, cod=T)
