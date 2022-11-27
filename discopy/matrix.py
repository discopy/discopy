"""
The category of matrices with the direct sum as monoidal product.

In this category, a box with domain ``n``
and codomain ``m`` represents an :math:`n \\times m` matrix.
The ``>>`` and ``<<`` operations correspond to matrix multiplication
and ``@`` operation corresponds to the direct sum of matrices:

.. math::

    \\mathbf{A} \\oplus \\mathbf{B}
    = \\begin{pmatrix} \\mathbf{A} & 0 \\\\ 0 & \\mathbf{B}  \\end{pmatrix}

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        block_diag

See also
--------
:class:`Matrix` is used to evaluate :class:`quantum.optics.Diagram`.

"""
from __future__ import annotations
import numpy as np

from discopy import cat, monoidal, messages
from discopy.cat import AxiomError, Composable
from discopy.monoidal import Whiskerable
from discopy.tensor import array2string
from discopy.utils import assert_isinstance


class Matrix(Composable, Whiskerable):
    """
    A matrix is a numpy array with integers as domain and codomain.

    Examples
    --------
    >>> m = Matrix([0, 1, 1, 0], 2, 2)
    >>> v = Matrix([0, 1], 1, 2)
    >>> v >> m >> v.dagger()
    Matrix([0], dom=1, cod=1)
    >>> m + m
    Matrix([0, 2, 2, 0], dom=2, cod=2)
    >>> assert m.then(m, m, m, m) == m >> m >> m >> m >> m

    The monoidal product for :py:class:`.Matrix` is the direct sum:

    >>> x = Matrix([2, 4], 2, 1)
    >>> x.array
    array([[2],
           [4]])
    >>> x @ x
    Matrix([2, 0, 4, 0, 0, 2, 0, 4], dom=4, cod=2)
    >>> (x @ x).array
    array([[2, 0],
           [4, 0],
           [0, 2],
           [0, 4]])
    """
    def __init__(self, array, dom: int, cod: int):
        assert_isinstance(dom, int)
        assert_isinstance(cod, int)
        self.dom, self.cod = dom, cod
        self.array = np.array(array).reshape((dom, cod))

    def __eq__(self, other):
        return isinstance(other, Matrix)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and np.all(self.array == other.array)

    def __repr__(self):
        return "Matrix({}, dom={}, cod={})".format(
            array2string(self.array.flatten()), self.dom, self.cod)

    def __str__(self):
        return repr(self)

    def then(self, other: Matrix = None, *others: Matrix):
        if others or other is None:
            return cat.Arrow.then(self, other, *others)
        assert_isinstance(other, Matrix)
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = np.matmul(self.array, other.array)
        return Matrix(array, self.dom, other.cod)

    def tensor(self, other: Matrix = None, *others: Matrix):
        if others or other is None:
            return monoidal.Diagram.tensor(self, other, *others)
        assert_isinstance(other, Matrix)
        dom, cod = self.dom + other.dom, self.cod + other.cod
        array = block_diag(self.array, other.array)
        return Matrix(array, dom, cod)

    def __add__(self, other):
        if other == 0:
            return self
        assert_isinstance(other, Matrix)
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Matrix(self.array + other.array, self.dom, self.cod)

    def __radd__(self, other):
        return self.__add__(other)

    def dagger(self):
        array = np.conjugate(np.transpose(self.array))
        return Matrix(array, self.cod, self.dom)

    @staticmethod
    def id(dom=0):
        return Matrix(np.identity(dom), dom, dom)

    @staticmethod
    def swap(left, right):
        if left == right == 1:
            return Matrix(np.array([0, 1, 1, 0], left @ right, left @ right))
        raise NotImplementedError


def block_diag(*arrs):
    """Compute the block diagonal of matrices, taken from scipy."""
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out
