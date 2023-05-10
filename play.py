from discopy.tensor import Box, Diagram, Tensor, Dim
from discopy.utils import assert_isinstance


print(Box)
print(Box[int])
print(Diagram)
print(Diagram[int])
print(Tensor)
print(Tensor[int])
assert_isinstance(1, int)
tt = Tensor[int]
d1 = Dim(1)
d2 = Dim(2)
t = tt([0, 1], d1, d2)
assert_isinstance(t, Tensor)
