"""
Render a :class:`typing.ParamSpec` by its name in type hints.

Sphinx 7.2 stringifies a :class:`typing.TypeVar` but not a
:class:`typing.ParamSpec`, and on Python 3.14 a signature such as
``Callable[Concatenate[type, P], T]`` evaluates to one, which crashed the
``typehints`` extension. Sphinx 7.4 renders it, so this extension retires
when the lockfile moves there.
"""

from typing import ParamSpec

from sphinx.ext.autodoc import typehints
from sphinx.util import typing

stringify = typing.stringify_annotation


def stringify_annotation(annotation, /, mode="fully-qualified-except-typing"):
    """ The name of a parameter specification, Sphinx's rendering else. """
    if isinstance(annotation, ParamSpec):
        return annotation.__name__
    return stringify(annotation, mode)


def setup(app):
    typing.stringify_annotation = stringify_annotation
    typehints.stringify_annotation = stringify_annotation

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
