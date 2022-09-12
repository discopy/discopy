"""
Implements the semantic category :py:class:`.Matrix`.

In this category, a box with domain :py:class:`PRO(n) <.monoidal.PRO>`
and codomain :py:class:`PRO(m) <.monoidal.PRO>` represents
an :math:`n \\times m` matrix.
The ``>>`` and ``<<`` operations correspond to matrix multiplication
and ``@`` operation corresponds to the direct sum of matrices:

.. math::

    \\mathbf{A} \\oplus \\mathbf{B}
    = \\begin{pmatrix} \\mathbf{A} & 0 \\\\ 0 & \\mathbf{B}  \\end{pmatrix}

Example
-------
>>> x = Matrix(PRO(2), PRO(1), [2, 4])
>>> x.array
array([[2],
       [4]])
>>> x @ x
Matrix(dom=PRO(4), cod=PRO(2), array=[2, 0, 4, 0, 0, 2, 0, 4])
>>> (x @ x).array
array([[2, 0],
       [4, 0],
       [0, 2],
       [0, 4]])

:py:class:`.Matrix` can be used to evaluate
:py:class:`.optics.Diagram` s from :py:mod:`.quantum.optics`.

"""

from discopy import messages, monoidal
from discopy.cat import AxiomError
from discopy.monoidal import PRO
from discopy.tensor import array2string
import numpy as np


class Matrix(monoidal.Box):
    """ Implements a matrix with dom, cod and numpy array.

    Examples
    --------
    >>> m = Matrix(PRO(2), PRO(2), [0, 1, 1, 0])
    >>> v = Matrix(PRO(1), PRO(2), [0, 1])
    >>> assert (str(v) == repr(v)
    ...                == 'Matrix(dom=PRO(1), cod=PRO(2), array=[0, 1])')
    >>> v >> m >> v.dagger()
    Matrix(dom=PRO(1), cod=PRO(1), array=[0])
    >>> m + m
    Matrix(dom=PRO(2), cod=PRO(2), array=[0, 2, 2, 0])
    >>> assert m.then(m, m, m, m) == m == m >> m >> m >> m >> m

    The monoidal product for :py:class:`.Matrix` is the direct sum:

    >>> x = Matrix(PRO(2), PRO(1), [2, 4])
    >>> x.array
    array([[2],
           [4]])
    >>> x @ x
    Matrix(dom=PRO(4), cod=PRO(2), array=[2, 0, 4, 0, 0, 2, 0, 4])
    >>> (x @ x).array
    array([[2, 0],
           [4, 0],
           [0, 2],
           [0, 4]])
    """
    def __init__(self, dom, cod, array):
        self._array = np.array(array).reshape((len(dom), len(cod)))
        super().__init__("O Tensor", dom, cod)

    @property
    def array(self):
        """ Numpy array. """
        return self._array

    def __repr__(self):
        return "Matrix(dom={!r}, cod={!r}, array={})".format(
            self.dom, self.cod, array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    def then(self, *others):
        from discopy import Sum
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other, = others
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = np.matmul(self.array, other.array)
        return Matrix(self.dom, other.cod, array)

    def tensor(self, *others):
        from discopy import Sum
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        array = block_diag(self.array, other.array)
        return Matrix(dom, cod, array)

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Matrix):
            raise TypeError(messages.type_err(Matrix, other))
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Matrix(self.dom, self.cod, self.array + other.array)

    def __radd__(self, other):
        return self.__add__(other)

    def dagger(self):
        array = np.conjugate(np.transpose(self.array))
        return Matrix(self.cod, self.dom, array)

    @staticmethod
    def id(dom=PRO()):
        return Matrix(dom, dom, np.identity(len(dom)))

    @staticmethod
    def swap(left, right):
        if left == PRO(1) and right == PRO(1):
            return Matrix(left @ right, left @ right, np.array([0, 1, 1, 0]))
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
