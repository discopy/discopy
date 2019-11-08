import numpy as np

class gates_to_numpy(dict):
    def __getitem__(self, g):
        if g.name == 'ket0' or g.name == 'bra0':
            return [0, 1]
        if g.name == 'ket1' or g.name == 'bra1':
            return [1, 0]
        elif g.name == 'S':
            return [1, 0,
                    0, 1j]
        elif g.name == 'T':
            return [np.exp(- 1j * np.pi / 2), 0,
                    0, np.exp(1j * np.pi / 2)]
        elif g.name == 'H':
            return 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
        elif g.name == 'X':
            return [0, 1,
                    1, 0]
        elif g.name == 'Y':
            return [0, -1j,
                    1j, 0]
        elif g.name == 'Z':
            return [1, 0,
                    0, -1]
        elif g.name[:2] in ['Rx', 'Rz']:
            theta = 2 * np.pi * float(g.params[0])
            if g.name[:2] == 'Rz':
                return [np.exp(-1j * theta / 2), 0,
                        0, np.exp(1j * theta / 2)]
            elif g.name[:2] == 'Rx':
                return [np.cos(theta / 2), -1j * np.sin(theta / 2),
                        -1j * np.sin(theta / 2), np.cos(theta / 2)]
        elif g.name == 'CX':
            return [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0]
        elif g.name == 'SWAP':
            return [1, 0, 0, 0,
                    0, 0, 1, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1]
        raise NotImplementedError

    def __repr__(self):
        return "GATES_TO_NUMPY"

GATES_TO_NUMPY = gates_to_numpy()
