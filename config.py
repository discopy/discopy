# -*- coding: utf-8 -*-

"""
If fast, checking axioms is disabled (approximately twice faster).
"""

VERSION = '0.1.2'

JAX = True

IGNORE = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

if JAX:
    import warnings
    for msg in IGNORE:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
else:
    import numpy as np
