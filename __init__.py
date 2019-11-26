from discopy import (
    cat, moncat, matrix, circuit, disco, config, _version)
from discopy.cat import (
    Ob, Arrow, Gen, Functor, Quiver)
from discopy.moncat import (
    Ty, Box, Diagram, MonoidalFunctor)
from discopy.matrix import (
    Dim, Matrix, MatrixFunctor)
from discopy.circuit import (
    PRO, Circuit, Gate, Bra, Ket, CircuitFunctor)
from discopy.disco import (
    Adjoint, Pregroup, Wire, Cup, Cap, Word, Model)

__version__ = config.VERSION
