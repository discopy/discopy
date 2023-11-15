from discopy.quantum.circuit import *
from discopy.quantum.gates import *

import sympy

x, y, z = sympy.symbols('x y z')
boxes = [
    Ket(0), Rx(0.552), Rz(x), Rx(0.917), Ket(0, 0, 0), H, H, H,
    CRz(0.18), CRz(y), CX, H, sqrt(2), Bra(0, 0), Ket(0),
    Rx(0.446), Rz(0.256), Rx(z), CX, H, sqrt(2), Bra(0, 0)]
offsets = [
    0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2, 3, 2, 0, 0, 0, 0, 0, 0, 1, 0]

# 0.6
# pick = Circuit(dom=qubit ** 0, cod=qubit, boxes=boxes, offsets=offsets)

# 1.1
pick = Circuit.decode(Ty(), zip(boxes, offsets))
