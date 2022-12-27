# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from __future__ import annotations

import json

from discopy import messages

from typing import Callable, Generic, Mapping, Iterable, TypeVar, Any,\
    Hashable,\
    Literal, cast, Union

from copy import deepcopy


KT = TypeVar('KT')
VT = TypeVar('VT')
V2T = TypeVar('V2T')


class DictOrCallable(Generic[KT, VT]):
    """ A Mapping or Callable object. """
    @property
    def is_dict(self):
        return isinstance(self.mapping, dict)

    def __init__(self, mapping: dict[KT, VT] | Callable[[KT], VT]) -> None:
        self.mapping: dict[KT, VT] | Callable[[KT], VT]
        self._inner_mapping: dict[KT, VT]
        if isinstance(mapping, DictOrCallable):
            self.mapping = deepcopy(mapping.mapping)
            self._inner_mapping = deepcopy(mapping._inner_mapping)
        else:
            self.mapping = mapping
            self._inner_mapping = {}

    def __getitem__(self, item: KT) -> VT:
        if isinstance(self.mapping, dict):
            return self.mapping[item]
        elif isinstance(item, Hashable)\
                and item in self._inner_mapping:
            return self._inner_mapping[item]
        else:
            return self.mapping(item)

    __call__ = __getitem__

    def __setitem__(self, key: KT, value: VT) -> None:
        """
        Sets the mapped value to a specified key.

        Note
        ----
        If the underlying structure is a function, a hidden dictionary is used
        to keep track of the set elements.

        Example
        -------
        >>> mapping = DictOrCallable({1: 'a', 2: 'b'})
        >>> mapping[1] = 'X'
        >>> print(mapping)
        {1: 'X', 2: 'b'}
        >>> callable_mapping = DictOrCallable(lambda x: x + x)
        >>> callable_mapping[1] = 'X'
        >>> assert callable_mapping(5) == 10
        >>> assert callable_mapping(1) == 'X'
        """
        if callable(setter := getattr(self.mapping, '__setitem__', None)):
            setter(key, value)
        else:
            self._inner_mapping[key] = value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DictOrCallable) and \
            self.mapping == other.mapping and \
            self._inner_mapping == other._inner_mapping

    def __repr__(self):
        return self.mapping.__repr__()

    def then(self, other: dict[VT, V2T] | Callable[[VT], V2T]) ->\
            DictOrCallable[KT, V2T]:
        """
        Returns the composition of the object with a dict or a Callable.

        Example
        -------
        >>> mapping = DictOrCallable({1: 'a', 2: 'b', 'c': 1})
        >>> print(mapping.then({'a': 1, 'b': 2, 'c': 3}))
        {1: 1, 2: 2}
        >>> print(mapping.then(lambda x: x * 3))
        {1: 'aaa', 2: 'bbb', 'c': 3}
        >>> mapping.then(mapping)
        {'c': 'a'}
        >>> assert mapping.then({}) == mapping.then(DictOrCallable({}))
        >>> DictOrCallable(lambda x: 2 * x).then(lambda x: x * 3)
        <function ...>

        """
        dict_1 = isinstance(other, dict) and other
        dict_2 = isinstance(other, DictOrCallable) and other.is_dict\
            and other.mapping
        dict_other = cast(Union[Literal[False], dict[VT, V2T]],
                          dict_1 if dict_2 is False else dict_2)
        if dict_other is False:
            return self._callable_then(cast(Callable[[VT], V2T], other))
        else:
            return self._dict_then(dict_other)

    def _dict_then(self, other: dict[VT, V2T]) -> DictOrCallable[KT, V2T]:
        if isinstance(self.mapping, dict):
            ret = DictOrCallable({
                x: other[y] for x, y in self.mapping.items() if y in other
            })
        else:
            ret = DictOrCallable(lambda x: other[self(x)])
        ret._inner_mapping =\
            {x: other[y] for x, y in self._inner_mapping.items()}
        return ret

    def _callable_then(self, other: Callable[[VT], V2T]) ->\
            DictOrCallable[KT, V2T]:
        if isinstance(self.mapping, dict):
            ret = DictOrCallable(
                {x: other(y) for x, y in self.mapping.items()}
            )
        else:
            ret = DictOrCallable(lambda x: other(self(x)))
        ret._inner_mapping =\
            {x: other(y) for x, y in self._inner_mapping.items()}
        return ret


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
