# -*- coding: utf-8 -*-

""" DisCoPy: the Python toolkit for computing with string diagrams. """

from importlib import metadata

from discopy import (
    cat,
    monoidal,
    braided,
    symmetric,
    markov,
    traced,
    closed,
    rigid,
    pivotal,
    ribbon,
    compact,
    frobenius,
    hypergraph,
    interaction,
    feedback,
    stream,
    python,
    matrix,
    tensor,
    quantum,
    grammar,
    drawing,
    utils,
    config,
    messages,
)

try:
    __version__ = metadata.version("discopy")
except metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__version_info__ = tuple(
    int(part) if part.isdigit() else part
    for part in __version__.replace("-", ".").split("."))
