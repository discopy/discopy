from discopy import cat, moncat, matrix, disco, circuit, config, _version
from discopy.cat import (
    Ob, Arrow, Gen, Functor, Quiver)
from discopy.moncat import (
    Ty, Box, Diagram, MonoidalFunctor)
from discopy.matrix import (
    Dim, Matrix, MatrixFunctor)
from discopy.disco import (
    Adjoint, Pregroup, Grammar, Wire, Cup, Cap, Word, Parse, Model)
from discopy.circuit import (
    PRO, Circuit, Gate, Bra, Ket, CircuitFunctor)

__version__ = _version.VERSION
