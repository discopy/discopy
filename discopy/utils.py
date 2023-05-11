# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from __future__ import annotations

import json
from functools import wraps
from typing import (
    Callable,
    Generic,
    Mapping,
    Iterable,
    TypeVar,
    Any,
    Hashable,
    Literal,
    cast,
    Union)

from discopy import messages


KT = TypeVar('KT')
VT = TypeVar('VT')
V2T = TypeVar('V2T')


class MappingOrCallable(Mapping[KT, VT]):
    """ A Mapping or Callable object. """
    def __class_getitem__(_, args: tuple[type, type]) -> type:
        source, target = args
        return Mapping[source, target] | Callable[[source], target]

    def __init__(self, mapping: MappingOrCallable[KT, VT]) -> None:
        while isinstance(mapping, MappingOrCallable):
            mapping = mapping.mapping
        self.mapping = mapping

    def __bool__(self) -> bool:
        return bool(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterable[KT]:
        for key in self.mapping:
            yield key

    def __getitem__(self, item: KT) -> VT:
        return self.mapping[item] if hasattr(self.mapping, "__getitem__")\
            else self.mapping(item)

    def __setitem__(self, key: KT, value: VT) -> None:
        """
        Sets the mapped value to a specified key.

        Example
        -------
        >>> mapping = MappingOrCallable({1: 'a', 2: 'b'})
        >>> mapping[1] = 'X'
        >>> print(mapping)
        {1: 'X', 2: 'b'}
        """
        self.mapping[key] = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MappingOrCallable):
            return self.mapping == other.mapping
        return self.mapping == other

    def __repr__(self):
        return repr(self.mapping)

    def then(self, other: MappingOrCallable[VT, V2T]
             ) -> MappingOrCallable[KT, V2T]:
        """
        Returns the composition of the object with a dict or a Callable.

        Example
        -------
        >>> mapping = MappingOrCallable({1: 'a'})
        >>> assert mapping.then({'a': 1}) == mapping.then(len) == {1: 1}
        """
        other = other if isinstance(other, Mapping)\
            else MappingOrCallable(other)
        if hasattr(self.mapping, "__iter__"):
            return MappingOrCallable({
                key: other[self[key]] for key in self.mapping})
        return MappingOrCallable(lambda key: other[self[key]])


class NamedGeneric(Generic[TypeVar('T')]):
    """
    A ``NamedGeneric`` is a ``Generic`` where the type parameter has a name.

    Parameters:
        attr : The name of the type parameter.

    Note
    ----
    In a standard ``Generic`` class, the type parameter disappears when the
    member of the class is instantiated, e.g.

    >>> assert list[int]([1, 2, 3])\\
    ...     == list[float]([1, 2, 3])\\
    ...     == [1, 2, 3]

    In a ``NamedGeneric``, the type parameter is attached to the members of the
    class so that we have access to it.

    Example
    -------

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class L(NamedGeneric["type_param"]):
    ...     inside: list
    >>> assert L[int]([1, 2, 3]).type_param == int
    >>> assert L[int]([1, 2, 3]) != L[float]([1, 2, 3])
    """
    def __class_getitem__(self, attr):
        class Result(Generic[TypeVar(attr)]):
            def __class_getitem__(cls, dtype: type, _cache=dict()):
                cls_attr = getattr(cls, attr, None)
                if cls_attr not in _cache or _cache[cls_attr] != cls:
                    _cache.clear()
                    _cache[cls_attr] = cls
                if dtype not in _cache:
                    origin = getattr(cls, "__origin__", cls)

                    class C(origin):
                        pass
                    C.__module__ = cls.__module__
                    C.__name__ = C.__qualname__ = \
                        f"{origin.__name__}[{dtype.__name__}]"
                    C.__origin__ = cls
                    setattr(C, attr, dtype)
                    _cache[dtype] = C
                return _cache[dtype]

            __name__ = __qualname__ = f"NamedGeneric['{attr}']"

        setattr(Result, attr, getattr(Result, attr, None))
        return Result


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
    return f"{cls.__module__.removeprefix('discopy.')}.{cls.__name__}"


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
    *modules, factory = tree['factory'].removeprefix('discopy.').split('.')
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


def assert_isinstance(object_, cls: type | tuple[type, ...]):
    """ Raise ``TypeError`` if ``object`` is not instance of ``cls``. """
    classes = cls if isinstance(cls, tuple) else (cls, )
    cls_name = ' | '.join(map(factory_name, classes))
    if not any(isinstance(object_, cls) for cls in classes):
        raise TypeError(messages.TYPE_ERROR.format(
            cls_name, factory_name(type(object_))))


def unbiased(binary_method):
    """
    Turn a biased method with signature (self, other) to an unbiased one, i.e.
    with signature (self, *others), see  the `nLab`_.

    .. _nLab: https://ncatlab.org/nlab/show/biased+definition
    """
    @wraps(binary_method)
    def method(self, *others):
        result = self
        for other in others:
            result = binary_method(result, other)
        return result
    return method
