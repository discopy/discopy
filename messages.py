# -*- coding: utf-8 -*-

"""
discopy error messages.
"""

IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]


def empty_name(got):
    """ Empty name error. """
    return "Expected non-empty name, got {}.".format(repr(got))


def type_err(expected, got):
    """ Type error. """
    return "Expected {}.{}, got {} of type {} instead.".format(
        expected.__module__, expected.__name__,
        repr(got), type(got).__name__)


def does_not_compose(left, right):
    """ Composition error. """
    return "{} does not compose with {}.".format(left, right)


def is_not_connected(diagram):
    """ Disconnected error. """
    return "{} is not connected.".format(str(diagram))


def boxes_and_offsets_must_have_same_len():
    """ Disconnected diagram error. """
    return "Boxes and offsets must have the same length."


def are_not_adjoints(left, right):
    """ Adjunction error. """
    return "{} and {} are not adjoints.".format(left, right)


def pivotal_not_implemented():
    """ Pivotal error. """
    return "Pivotal categories are not implemented."


def cup_vs_cups(left, right):
    """ Simple type error. """
    return "Cup can only witness adjunctions between simple types. "\
           "Use Diagram.cups({}, {}) instead.".format(left, right)


def cap_vs_caps(left, right):
    """ Simple type error. """
    return cup_vs_cups(left, right).replace('up', 'ap')


def cannot_add(left, right):
    """ Addition error. """
    return "Cannot add {} and {}.".format(left, right)


def expected_pregroup():
    """ pregroup.draw error. """
    return "Expected a pregroup diagram, use diagram.draw() instead."


def expected_input_length(function, values):
    return "Expected input of length {}, got {} instead.".format(
        len(function.dom), len(values))
