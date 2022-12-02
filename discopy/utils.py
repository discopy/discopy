# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from __future__ import annotations

import json

from collections.abc import Mapping, Iterable

from discopy import messages


def product(xs: list[int], unit=1):
    """ The product of a list of integers ``xs``. """
    return unit if not xs else product(xs[1:], unit * xs[0])


def factory_name(cls: type) -> str:
    """ Returns a string describing a DisCoPy class. """
    return "{}.{}".format(
        cls.__module__.removeprefix("discopy."), cls.__name__)


def from_tree(tree):
    """ Decodes a tree as a DisCoPy object. """
    *modules, factory = tree['factory'].split('.')
    import discopy
    module = discopy
    for attr in modules:
        module = getattr(module, attr)
    return getattr(module, factory).from_tree(tree)


def dumps(obj):
    """
    Serialise a DisCoPy object as JSON.

    >>> from pprint import PrettyPrinter
    >>> pprint = PrettyPrinter(indent=4, width=60).pprint
    >>> from discopy.cat import Box, Ob
    >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {'Alice': 1}])
    >>> d = f >> f[::-1]
    >>> assert loads(dumps(d)) == d
    >>> pprint(json.loads(dumps(d)))
    {   'cod': {'factory': 'cat.Ob', 'name': 'x'},
        'dom': {'factory': 'cat.Ob', 'name': 'x'},
        'factory': 'cat.Arrow',
        'inside': [   {   'cod': {   'factory': 'cat.Ob',
                                     'name': 'y'},
                          'data': [42, {'Alice': 1}],
                          'dom': {   'factory': 'cat.Ob',
                                     'name': 'x'},
                          'factory': 'cat.Box',
                          'name': 'f'},
                      {   'cod': {   'factory': 'cat.Ob',
                                     'name': 'x'},
                          'data': [42, {'Alice': 1}],
                          'dom': {   'factory': 'cat.Ob',
                                     'name': 'y'},
                          'factory': 'cat.Box',
                          'is_dagger': True,
                          'name': 'f'}]}
    """
    return json.dumps(obj.to_tree())


def loads(raw):
    """ Loads a serialised DisCoPy object. """
    obj = json.loads(raw)
    if isinstance(obj, list):
        return [from_tree(o) for o in obj]
    return from_tree(obj)


def rmap(func, data):
    """
    Apply :code:`func` recursively to :code:`data`.

    Examples
    --------
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
    import urllib.request as urllib
    import zipfile

    fd, _ = urllib.urlretrieve(url)
    zip_file = zipfile.ZipFile(fd, 'r')
    first_file = zip_file.namelist()[0]
    with zip_file.open(first_file) as f:
        diagrams = loads(f.read())

    return diagrams


def assert_isinstance(object, cls: type | tuple[type, ...]):
    classes = cls if isinstance(cls, tuple) else (cls, )
    cls_name = ' or '.join(map(factory_name, classes))
    if not any(isinstance(object, cls) for cls in classes):
        raise TypeError(messages.TYPE_ERROR.format(
            cls_name, factory_name(type(object))))


class BinaryBoxConstructor:
    """ Box constructor with left and right as input. """
    def __init__(self, left, right):
        self.left, self.right = left, right

    def __repr__(self):
        return "{}({}, {})".format(
            factory_name(type(self)), repr(self.left), repr(self.right))

    def to_tree(self):
        left, right = self.left.to_tree(), self.right.to_tree()
        return dict(factory=factory_name(type(self)), left=left, right=right)

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, (tree['left'], tree['right'])))
