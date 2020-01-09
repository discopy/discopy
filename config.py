# -*- coding: utf-8 -*-

"""
discopy package configuration.
"""

VERSION = '0.1.5'

IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]


class Msg:
    """ Error messages. """
    @staticmethod
    def type_err(expected, got):
        """ Type error. """
        return "Expected {}.{}, got {} of type {} instead.".format(
            expected.__module__, expected.__name__,
            repr(got), type(got).__name__)

    @staticmethod
    def does_not_compose(left, right):
        """ Composition error. """
        return "{} does not compose with {}.".format(str(left), str(right))

    @staticmethod
    def is_not_connected(diagram):
        """ Disconnected error. """
        return "{} is not connected.".format(str(diagram))

    @staticmethod
    def boxes_and_offsets_must_have_same_len():
        """ Disconnected diagram error. """
        return "Boxes and offsets must have the same length."

    @staticmethod
    def are_not_adjoints(left, right):
        """ Adjunction error. """
        return "{} and {} are not adjoints.".format(left, right)

    @staticmethod
    def pivotal_not_implemented():
        """ Pivotal error. """
        return "Pivotal categories are not implemented."

    @staticmethod
    def cup_vs_cups(left, right):
        """ Simple type error. """
        return "Cup can only witness adjunctions between simple types. "\
               "Use Diagram.cups({}, {}) instead.".format(left, right)

    @staticmethod
    def cap_vs_caps(left, right):
        """ Simple type error. """
        return Msg.cup_vs_cups(left, right).replace('up', 'ap')

    @staticmethod
    def cannot_add(left, right):
        """ Addition error. """
        return "Cannot add {} and {}.".format(left, right)
