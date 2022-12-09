# -*- coding: utf-8 -*-

"""
DisCoPy utility functions.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    from_tree
    dumps
    loads
"""

from __future__ import annotations

import json

from collections.abc import Mapping, Iterable

from discopy import messages

class Dict:
    """ dict-like object from callable. """
    def __init__(self, func: Callable):
        self.func = func

    def __getitem__(self, key):
        return self.func(key)


def product(xs: list, unit=1):
    """
    The left-fold product of a ``unit`` with list of ``xs``.

    Example
    -------
    >>> assert product([1, 2, 3]) == 6
    >>> assert product([1, 2, 3], unit=[42]) == 6 * [42]
    """
    return unit if not xs else product(xs[1:], unit * xs[0])


def factory_name(cls: type) -> str:
    """
    Returns a string describing a DisCoPy class.

    Example
    -------
    >>> from discopy.grammar.pregroup import Word
    >>> assert factory_name(Word) == "grammar.pregroup.Word"
    """
    return "{}.{}".format(
        cls.__module__.removeprefix("discopy."), cls.__name__)


def from_tree(tree: dict):
    """
    Import DisCoPy and decode a serialised object.

    Parameters:
        tree : The serialisation of a DisCoPy object.

    Example
    -------
    >>> tree = {'factory': 'cat.Arrow',
    ...         'inside': [   {   'factory': 'cat.Box',
    ...                           'name': 'f',
    ...                           'dom': {'factory': 'cat.Ob', 'name': 'x'},
    ...                           'cod': {'factory': 'cat.Ob', 'name': 'y'},
    ...                           'data': 42},
    ...                       {   'factory': 'cat.Box',
    ...                           'name': 'f',
    ...                           'dom': {'factory': 'cat.Ob', 'name': 'y'},
    ...                           'cod': {'factory': 'cat.Ob', 'name': 'x'},
    ...                           'is_dagger': True,
    ...                           'data': 42}],
    ...         'dom': {'factory': 'cat.Ob', 'name': 'x'},
    ...         'cod': {'factory': 'cat.Ob', 'name': 'x'}}

    >>> from discopy.cat import Box
    >>> f = Box('f', 'x', 'y', data=42)
    >>> assert from_tree(tree) == f >> f[::-1]
    """
    *modules, factory = tree['factory'].split('.')
    import discopy
    module = discopy
    for attr in modules:
        module = getattr(module, attr)
    return getattr(module, factory).from_tree(tree)


def dumps(obj, **kwargs):
    """
    Serialise a DisCoPy object as JSON.

    Parameters:
        obj : The DisCoPy object to serialise.
        kwargs : Passed to ``json.dumps``.

    Example
    -------
    >>> from discopy.cat import Box, Id
    >>> f = Box('f', 'x', 'y', data=42)
    >>> print(dumps(f[::-1] >> Id('x'), indent=4))
    {
        "factory": "cat.Arrow",
        "inside": [
            {
                "factory": "cat.Box",
                "name": "f",
                "dom": {
                    "factory": "cat.Ob",
                    "name": "y"
                },
                "cod": {
                    "factory": "cat.Ob",
                    "name": "x"
                },
                "is_dagger": true,
                "data": 42
            }
        ],
        "dom": {
            "factory": "cat.Ob",
            "name": "y"
        },
        "cod": {
            "factory": "cat.Ob",
            "name": "x"
        }
    }
    """
    return json.dumps(obj.to_tree(), **kwargs)


def loads(raw):
    """
    Loads a serialised DisCoPy object.

    Example
    -------
    >>> raw = '{"factory": "cat.Ob", "name": "x"}'
    >>> from discopy.cat import Ob
    >>> assert loads(raw) == Ob('x')
    >>> assert dumps(loads(raw)) == raw
    >>> assert loads(dumps(Ob('x'))) == Ob('x')
    """
    obj = json.loads(raw)
    if isinstance(obj, list):
        return [from_tree(o) for o in obj]
    return from_tree(obj)


def rmap(func, data):
    """
    Apply :code:`func` recursively to :code:`data`.

    Example
    -------
    >>> data = {'A': [0, 1, 2], 'B': ({'C': 3, 'D': [4, 5, 6]}, {7, 8, 9})}
    >>> rmap(lambda x: x + 1, data)
    {'A': [1, 2, 3], 'B': ({'C': 4, 'D': [5, 6, 7]}, {8, 9, 10})}
    """
    if isinstance(data, Mapping):
        return {key: rmap(func, value) for key, value in data.items()}
    if isinstance(data, Iterable):
        return type(data)([rmap(func, elem) for elem in data])
    return func(data)


def rsubs(data, *args):
    """ Substitute recursively along nested data. """
    from sympy import lambdify
    if isinstance(args, Iterable) and not isinstance(args[0], Iterable):
        args = (args, )
    keys, values = zip(*args)
    return rmap(lambda x: lambdify(keys, x)(*values), data)


def load_corpus(url):
    """ Load a corpus hosted at a given ``url``. """
    import urllib.request as urllib
    import zipfile

    fd, _ = urllib.urlretrieve(url)
    zip_file = zipfile.ZipFile(fd, 'r')
    first_file = zip_file.namelist()[0]
    with zip_file.open(first_file) as f:
        return loads(f.read())


def assert_isinstance(object, cls: type | tuple[type, ...]):
    """ Raise ``TypeError`` if ``object`` is not instance of ``cls``. """
    classes = cls if isinstance(cls, tuple) else (cls, )
    cls_name = ' | '.join(map(factory_name, classes))
    if not any(isinstance(object, cls) for cls in classes):
        raise TypeError(messages.TYPE_ERROR.format(
            cls_name, factory_name(type(object))))
