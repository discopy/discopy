from discopy.tensor import Tensor, Dim
import numpy as np

arr = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape((2, 2, 2))

# 0.6
# pick = Tensor(Dim(2, 2), Dim(2), arr)

# main
pick = Tensor(arr, Dim(2, 2), Dim(2))
