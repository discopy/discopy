from discopy import Sum, messages, monoidal, PRO
from discopy.cat import AxiomError
from discopy.tensor import array2string
from scipy.linalg import block_diag
import numpy as np


class Matrix(monoidal.Box):

    def __init__(self, dom, cod, array):
        self._array = np.array(array).reshape((len(dom), len(cod)))
        super().__init__("O Tensor", dom, cod)

    @property
    def array(self):
        """ Numpy array. """
        return self._array

    def __repr__(self):
        return "Matrix(dom={}, cod={}, array={})".format(
            self.dom, self.cod, array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    def then(self, *others):
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
