import quimb.tensor as qtn
import numpy as np

from discopy.quantum import *


def to_tn(diagram):
    inputs = [
            qtn.COPY_tensor(
                d=getattr(dim, 'dim', dim),
                inds=(f'inp{i}', f'inp{i}_end')
            ) for i, dim in enumerate(diagram.dom.inside)]
    tensors = inputs[:]
    scan = [(t, 1) for t in inputs]

    for i, (box, off) in enumerate(zip(diagram.boxes, diagram.offsets)):
        if isinstance(box, Swap):
            scan[off], scan[off + 1] = scan[off + 1], scan[off]
            continue

        in_inds = [f't{i}_i{j}' for j in range(len(box.dom))]
        out_inds = [f't{i}_o{j}' for j in range(len(box.cod))]
        t = qtn.Tensor(
            data=box.array,
            inds=in_inds + out_inds,
        )
        tensors.append(t)
        for j in range(len(box.dom)):
            other_t, other_ind = scan[off + j]
            qtn.connect(other_t, t, other_ind, j)

        scan[off:off + len(box.dom)] = [
            (t, len(box.dom) + ind) for ind in range(len(out_inds))
        ]

    tensor_net = qtn.TensorNetwork(tensors)
    tensor_net.reindex({
        t.inds[j]: f'out{i}' for i, (t, j) in enumerate(scan)
    }, inplace=True)

    return tensor_net


pure_circuits = [
    H >> X >> Y >> Z,
    CX >> H @ Rz(0.5),
    CRz(0.123) >> Z @ Z,
    CX >> H @ qubit >> Bra(0, 0),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]).l.dagger(),
    Circuit.permutation([1, 2, 0]) >> H @ Z @ X,
    Circuit.permutation([1,2,0]),
]


def check(c):
    r = to_tn(c).contract()
    a1 = r.data.transpose(*np.argsort(r.inds))
    a2 = c.eval().array

    assert np.allclose(a1, a2), c.draw()


for i, c in enumerate(pure_circuits):
    check(c)

print("All tests passed!")
