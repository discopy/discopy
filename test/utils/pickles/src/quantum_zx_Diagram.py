from discopy.quantum.zx import Z, Id, SWAP, X


pick = Z(1, 2, phase=1) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
