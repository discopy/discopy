from discopy import messages, monoidal
from discopy.cat import AxiomError
from discopy.monoidal import PRO
from discopy.tensor import array2string
from scipy.linalg import block_diag
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

    The monoidal product for Matrix is the direct product:

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
