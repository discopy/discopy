# -*- coding: utf-8 -*-

"""
Loading a checkpoint trained before the refactor of ``discopy.neural``.

The refactor changed no weights and no arithmetic, and for three of the
four models it changed no names either: a checkpoint of the factor-graph
model, of the recursion model or of its ACT variant loads with
``strict=True`` and no translation at all.

One model needs a rename.  The clique cell used to be an ``RRNCell`` whose
pairwise message function was called ``message``; it is now a
:class:`discopy.neural.cells.Site` whose encoder is called ``encode``,
because that is what the same layer is called in every other site.  Same
shapes, same values, same order -- three keys with a different prefix.

    from migration import load
    load(model, "artifacts/quick-best-rrn-seed0.pt")

:func:`rename` is the map itself, and :func:`load` applies it to a
checkpoint saved either as a bare ``state_dict`` or as the study's
``{"state_dict": ..., "history": ..., "meta": ...}`` record.

Example
-------
>>> rename("rrn", "cell.message.0.weight")
'cell.encode.0.weight'
>>> rename("rrn", "cells.networks.0.message.4.bias")
'cells.networks.0.encode.4.bias'
>>> rename("goi", "cell.encode.0.weight")
'cell.encode.0.weight'
>>> rename("rrn", "readout.bias")
'readout.bias'
"""

from __future__ import annotations

#: Per model, the substrings a checkpoint's keys are rewritten by.  A model
#: with no entry needs no translation.
RENAMES = {
    "rrn": ((".message.", ".encode."), ),
}


def rename(model: str, key: str) -> str:
    """
    The name a pre-refactor checkpoint key has after the refactor.

    Parameters:
        model : ``"goi"``, ``"rrn"``, ``"trm"`` or ``"act"``.
        key : The key as it appears in the old ``state_dict``.
    """
    for old, new in RENAMES.get(model, ()):
        key = key.replace(old, new)
    return key


def translate(model: str, state_dict: dict) -> dict:
    """
    A whole ``state_dict``, renamed.

    Parameters:
        model : The name of the model the checkpoint was trained as.
        state_dict : The stored weights.
    """
    return {rename(model, key): value for key, value in state_dict.items()}


def load(module, path, model: str = None, strict: bool = True):
    """
    Load a pre-refactor checkpoint into a model.

    Parameters:
        module : The model to load into.
        path : The checkpoint, a bare ``state_dict`` or the study's record.
        model : The name it was trained as, guessed from the keys by
                default.
        strict : Whether to require every key to match, as it should.

    Returns:
        Whatever ``torch.load`` read, so that the caller can still see the
        history and metadata beside the weights.
    """
    import torch
    stored = torch.load(path, map_location="cpu", weights_only=False)
    weights = stored["state_dict"] if isinstance(stored, dict) \
        and "state_dict" in stored else stored
    if model is None:
        model = "rrn" if any(".message." in key for key in weights) else "goi"
    module.load_state_dict(translate(model, weights), strict=strict)
    return stored
