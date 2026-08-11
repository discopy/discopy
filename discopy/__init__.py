# -*- coding: utf-8 -*-

""" DisCoPy: the Python toolkit for computing with string diagrams. """

import doctest

from discopy import (
    abc,
    cat,
    monoidal,
    braided,
    symmetric,
    traced,
    biclosed,
    rigid,
    pivotal,
    ribbon,
    compact,
    markov,
    closed,
    frobenius,
    hypergraph,
    cmap,
    interaction,
    feedback,
    stream,
    python,
    matrix,
    tensor,
    hopf,
    quantum,
    grammar,
    drawing,
    utils,
    config,
    messages,
)

from discopy.version import (
    version as __version__,
    version_tuple as __version_info__
)

# A docstring example that needs an optional backend says so with `+EXTRA`.
# Registering the name keeps those docstrings parseable by plain doctest;
# acting on it is the job of `--skip-extra`, see discopy/pytest_plugin.py.
doctest.register_optionflag("EXTRA")
