# -*- coding: utf-8 -*-

"""
The sudoku encoder and decoder -- which turn out not to be sudoku's.

Both moved to :mod:`core.heads`: a class-token embedding with a blank and
a linear readout with the fill-in-the-blanks decode rule are the
*family*'s two ends, shared by any benchmark whose input is a partial
solution.  This module remains as the import path the task has always
had; the classes, and hence the checkpoint keys ``embedding.weight`` and
``readout.weight``, are unchanged.
"""

from __future__ import annotations

from core.heads import Decoder, Encoder  # noqa: F401 -- re-exported

__all__ = ["Decoder", "Encoder"]
