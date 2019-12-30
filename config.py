# -*- coding: utf-8 -*-

"""
discopy package configuration.
"""

VERSION = '0.1.4'

IGNORE = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]

try:
    import warnings
    for msg in IGNORE:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np
except ImportError:
    import numpy as np
