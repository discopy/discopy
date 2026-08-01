# -*- coding: utf-8 -*-

"""
The two ends of a sudoku solver: what turns a puzzle into messages, and
what turns messages back into digits.

Everything between them -- the wiring, the cells, the schedule -- is the
generic engine of :mod:`discopy.neural`.  These two are the only modules
that know the task is sudoku, and even they know very little: a puzzle is
a grid of class indices with ``0`` for a blank, and a prediction is a class
per cell with the givens written back over it.

Both subclass the torch module they *are*, rather than wrapping it, so a
checkpoint's keys are the plain ``embedding.weight`` and
``readout.weight`` they have always been.
"""

from __future__ import annotations

import torch


class Encoder(torch.nn.Embedding):
    """
    The clue embedding: one vector per digit, plus one for a blank.

    Parameters:
        n_classes : The number of digits a cell can take.
        dim : The width of the embedding, i.e. of a clue wire.

    Example
    -------
    >>> Encoder(9, 4)(torch.zeros(2, 81, dtype=torch.long)).shape
    torch.Size([2, 81, 4])
    """
    def __init__(self, n_classes: int, dim: int):
        super().__init__(n_classes + 1, dim)


class Decoder(torch.nn.Linear):
    """
    The readout: class logits from whatever the engine decodes, plus the
    fill-in-the-blanks rule that turns them into a solution.

    Parameters:
        dim : The width of what is decoded -- a latent state or an answer
              embedding, depending on the schedule.
        n_classes : The number of digits a cell can take.

    Example
    -------
    >>> logits = torch.zeros(1, 3, 9)
    >>> logits[0, 0, 4] = 1.0
    >>> Decoder.decode(logits, torch.tensor([[0, 7, 0]]))
    tensor([[5, 7, 1]])
    """
    def __init__(self, dim: int, n_classes: int):
        super().__init__(dim, n_classes)

    @staticmethod
    def decode(logits, clues):
        """
        Argmax classes with the clues written back over the predictions: a
        given is never something to predict.

        Parameters:
            logits : Class logits of shape ``(batch, cells, classes)``.
            clues : The puzzles, ``0`` for a blank.
        """
        predicted = logits.argmax(-1) + 1
        return torch.where(clues > 0, clues, predicted)
