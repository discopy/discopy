# -*- coding: utf-8 -*-

"""
discopy error messages.
"""


def type_err(expected, got):
    """ Type error. """
    return "Expected {}.{}, got {} of type {}.{} instead.".format(
        expected.__module__, expected.__name__,
        repr(got), type(got).__module__, type(got).__name__)


def does_not_compose(left, right):
    """ Composition error. """
    return "{} does not compose with {}.".format(left, right)


def is_not_connected(diagram):
    """ Disconnected error. """
    return "{} is not connected.".format(str(diagram))


def boxes_and_offsets_must_have_same_len():
    """ Disconnected diagram error. """
    return "Boxes and offsets must have the same length."


def no_winding_number_for_complex_types():
    """ No winding number for complex types. """
    return "Only atomic types have a winding number."


def are_not_adjoints(left, right):
    """ Adjunction error. """
    return "{} and {} are not adjoints.".format(left, right)


def cup_vs_cups(left, right):
    """ Simple type error. """
    return "Cup can only witness adjunctions between simple types. "\
           "Use Diagram.cups({}, {}) instead.".format(left, right)


def cap_vs_caps(left, right):
    """ Simple type error. """
    return cup_vs_cups(left, right).replace('up', 'ap')


def swap_vs_swaps(left, right):
    """ Simple type error. """
    return cup_vs_cups(left, right).replace("adjunctions", "symmetry")\
        .replace("Cup", "Swap").replace("cups", "swap")


def cannot_add(left, right):
    """ Addition error. """
    return "Cannot add {} and {}.".format(left, right)


def missing_types_for_empty_sum():
    """ Empty sum needs types. """
    return "Empty sum needs a domain and codomain."


def expected_pregroup():
    """ pregroup.draw error. """
    return "Expected a pregroup diagram of shape"\
           "`word @ ... @ word >> cups_and_swaps`,"\
           " use diagram.draw() instead."


def expected_input_length(function, values):
    """ Unexpected input length error. """
    return "Expected input of length {}, got {} instead.".format(
        len(function.dom), len(values))
