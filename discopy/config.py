# -*- coding: utf-8 -*-

""" Discopy configuration. """

IMPORT_JAX = False
NUMPY_THRESHOLD = 16
DEFAULT_DTYPE = int, 32
DEFAULT_BACKEND = 'numpy'
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]
