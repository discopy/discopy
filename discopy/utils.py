# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from collections.abc import Mapping, Iterable

import json


def factory_name(obj):
    """ Returns a string describing a DisCoPy object. """
    return '{}.{}'.format(type(obj).__module__, type(obj).__name__)


def from_tree(tree):
    """ Decodes a tree as a DisCoPy object. """
    package, *modules, factory = tree['factory'].split('.')
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
    {   'boxes': [   {   'cod': {   'factory': 'discopy.cat.Ob',
                                    'name': 'y'},
                         'data': [42, {'Alice': 1}],
                         'dom': {   'factory': 'discopy.cat.Ob',
                                    'name': 'x'},
                         'factory': 'discopy.cat.Box',
                         'name': 'f'},
                     {   'cod': {   'factory': 'discopy.cat.Ob',
                                    'name': 'x'},
                         'data': [42, {'Alice': 1}],
                         'dom': {   'factory': 'discopy.cat.Ob',
                                    'name': 'y'},
                         'factory': 'discopy.cat.Box',
                         'is_dagger': True,
                         'name': 'f'}],
        'cod': {'factory': 'discopy.cat.Ob', 'name': 'x'},
        'dom': {'factory': 'discopy.cat.Ob', 'name': 'x'},
        'factory': 'discopy.cat.Arrow'}
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
    return rmap(lambda x: getattr(x, "subs", lambda *_: x)(*args), data)


def load_corpus(url):
    import urllib.request as urllib
    import zipfile

    fd, _ = urllib.urlretrieve(url)
    zip_file = zipfile.ZipFile(fd, 'r')
    first_file = zip_file.namelist()[0]
    with zip_file.open(first_file) as f:
        diagrams = loads(f.read())

    return diagrams
