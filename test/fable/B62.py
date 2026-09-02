"""B62: the class NamedGeneric["attr"] returns does not subclass NamedGeneric, so unpickling loses dtype and category (discopy/abc.py:639).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pickle

from discopy import compact, symmetric, tensor
from discopy.matrix import Matrix
from discopy.tensor import Dim


def test_b62_matrix_dtype_survives_pickle():
    M = Matrix[int]([1, 0, 0, 1], 2, 2)
    L = pickle.loads(pickle.dumps(M))
    assert L.dtype is int and L == M


def test_b62_hypergraph_category_survives_pickle():
    f = symmetric.Box('f', symmetric.Ty('x'), symmetric.Ty('y'))
    L = pickle.loads(pickle.dumps(f.to_hypergraph()))
    assert L.category is symmetric.Diagram
    assert L.to_diagram() == f


def test_b62_cmap_category_survives_pickle():
    f = compact.Box('f', compact.Ty('x'), compact.Ty('y'))
    L = pickle.loads(pickle.dumps(compact.CMap.from_box(f)))
    assert L.category is compact.Diagram
    assert L.to_diagram() == f


def test_b62_tensor_box_dtype_survives_pickle_control():
    """Passing control: tensor.Box calls NamedGeneric.__setstate__ by hand."""
    box = tensor.Box[int]('f', Dim(2), Dim(2), [1, 0, 0, 1])
    assert pickle.loads(pickle.dumps(box)).dtype is int
