from discopy import *
from discopy.circuit import *


class QuantumMap(Box):
    def __init__(self, dom, cod, array):
        self.array = array
        super().__init__(array, dom, cod,
                         data=Tensor(dom[::-1] @ dom, cod[::-1] @ cod, array))

    def __repr__(self):
        return "QuantumMap(dom={}, cod={}, array={})".format(
            self.dom, self.cod, self.array.flatten())

    def __eq__(self, other):
        return isinstance(other, QuantumMap)\
            and self.dom == other.dom and self.cod == other.cod\
            and self.data == other.data

    def tensor(self, other):
        dom, cod, tensor = self.dom @ other.dom, self.cod @ other.cod,\
            Tensor.id(self.dom[::-1]) @ Tensor.swap(other.dom[::-1], self.dom)\
            @ Tensor.id(other.dom) >> self.tensor @ other.tensor\
            >> Tensor.id(self.cod[::-1]) @ swap(self.cod, other.cod[::-1])\
            @ Tensor.id(other.cod)
        return QuantumMap(dom, cod, tensor.array)

    def then(self, other):
        data = self.data >> other.data
        return QuantumMap(self.dom, other.cod, data.array)

    @staticmethod
    def id(dim):
        data = Tensor.id(dim[::-1]) @ Tensor.id(dim)
        return QuantumMap(dim, dim, data.array)


class PureMap(QuantumMap):
    def __init__(self, pure):
        super().__init__(pure.dom, pure.cod,
                         (pure.dagger().transpose() @ pure).array)


class Discard(QuantumMap):
    def __init__(self, dim):
        super().__init__(dim, Dim(1), Tensor.id(dim).array)
