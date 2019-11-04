import numpy as np
from moncat import Ob, Ty, Box, Diagram, MonoidalFunctor


class NumpyFunctor(MonoidalFunctor):
    def __call__(self, d):
        if isinstance(d, Ob):
            return int(self.ob[d])
        elif isinstance(d, Ty):
            return tuple(self.ob[x] for x in d)
        elif isinstance(d, Box):
            arr = np.array(self.ar[d.name])
            if d._dagger:
                arr = arr.reshape(self(d.cod) + self(d.dom))
                return np.moveaxis(arr, range(len(arr.shape)),
                    [i + len(d.dom) if i < len(d.cod) else
                     i - len(d.cod) for i in range(len(arr.shape))])
            else:
                return arr.reshape(self(d.dom) + self(d.cod))
        arr = 1
        for x in d.dom:
            arr = np.tensordot(arr, np.identity(self(x)), 0)
        arr = np.moveaxis(arr,
            [2 * i for i in range(len(d.dom))],
            [i for i in range(len(d.dom))])  # bureaucracy!

        for f, n in d:
            source = range(len(d.dom) + n, len(d.dom) + n + len(f.dom))
            target = range(len(f.dom))
            arr = np.tensordot(arr, self(f), (source, target))

            source = range(len(arr.shape) - len(f.cod), len(arr.shape))
            target = range(len(d.dom) + n, len(d.dom) + n +len(f.cod))
            arr = np.moveaxis(arr, source, target)  # more bureaucracy!
        return arr
