# -*- coding: utf-8 -*-

from discopy import cat, moncat, matrix, circuit, gates, disco, config
from discopy.cat import Ob, Arrow, Gen, Functor, Quiver
from discopy.moncat import MonoidalFunctor
from discopy.matrix import Dim, Matrix, MatrixFunctor
from discopy.circuit import PRO, Circuit, CircuitFunctor
from discopy.gates import Gate, Bra, Ket
from discopy.pregroup import Adjoint, Pregroup, Box, Diagram, Wire, Cup, Cap
from discopy.disco import Word, Model, CircuitModel

__version__ = config.VERSION
