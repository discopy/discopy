# -*- coding: utf-8 -*-

""" DisCoPy utility functions. """

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import (
    Callable,
    Generic,
    Mapping,
    Iterable,
    TypeVar,
    Any,
    Collection,
    Type,
    Optional,
    TYPE_CHECKING,
)

from matplotlib.testing.compare import compare_images
from networkx import Graph, connected_components

from discopy import messages
from discopy.config import DRAWING_DEFAULT

if TYPE_CHECKING:
    from discopy.monoidal import Ty, Diagram

KT = TypeVar('KT')
VT = TypeVar('VT')
V2T = TypeVar('V2T')


class MappingOrCallable(Mapping[KT, VT]):
    """
    A Mapping or Callable object.

    Example
    -------
    T = MappingOrCallable[int, int]
    f = T(lambda x: x + 1)
    g = T({0: 1})
    assert f[0] == g[0] == 1
    assert len(g) == 1
    assert list(g) == [0]
    """
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
    _cache = dict()

    def __class_getitem__(_, attributes):
        if not isinstance(attributes, tuple):
            attributes = (attributes,)

        class Result(Generic[TypeVar(attributes)]):
            def __class_getitem__(cls, values):
                if hasattr(cls, "__is_named_generic__"):
                    cls = cls.__bases__[0]
                values = values if isinstance(values, tuple) else (values,)
                cls_values = tuple(
                    getattr(cls, attr, None) for attr in attributes)
                if cls not in NamedGeneric._cache:
                    NamedGeneric._cache[cls] = {cls_values: cls}
                if values not in NamedGeneric._cache[cls]:
                    origin = getattr(cls, "__origin__", cls)

                    class C(origin):
                        __is_named_generic__ = True

                        # We need this to fix pickling of nested classes
                        # https://stackoverflow.com/questions/1947904/how-can-i-pickle-a-dynamically-created-nested-class-in-python
                        def __reduce__(self):
                            func, args, data = super().__reduce__()
                            # Check if class name is of the form:
                            # *ClassName*[*type*]
                            if '[' in args[0].__name__:
                                args = (origin, ) + args[1:]
                                data |= {"__class_getitem__values__": values}
                            return func, args, data

                    C.__module__ = origin.__module__
                    names = [getattr(v, "__name__", str(v)) for v in values]
                    C.__name__ = C.__qualname__ = origin.__name__\
                        + f"[{', '.join(names)}]"
                    C.__origin__ = cls
                    for attr, value in zip(attributes, values):
                        setattr(C, attr, value)
                    NamedGeneric._cache[cls][values] = C
                return NamedGeneric._cache[cls][values]

            __name__ = __qualname__\
                = f"NamedGeneric[{', '.join(map(repr, attributes))}]"

        for attr in attributes:
            setattr(Result, attr, getattr(Result, attr, None))
        return Result

    def __setstate__(self, state):
        if "__class_getitem__values__" in state:
            new_cls = self.__class__[state["__class_getitem__values__"]]
            self.__class__ = new_cls


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
    module = cls.__module__.removeprefix('discopy.')
    return f"{module}.{cls.__name__}".removeprefix('builtins.')


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
    with signature (self, *others), see the `nLab`_.

    .. _nLab: https://ncatlab.org/nlab/show/biased+definition
    """
    @wraps(binary_method)
    def method(self, *others):
        result = self
        for other in others:
            result = binary_method(result, other)
        return result
    return method


Pushout = tuple[dict[int, int], dict[int, int]]


def pushout(
        left: int, right: int,
        left_boundary: Collection[int], right_boundary: Collection[int]) \
        -> Pushout:
    """
    Computes the pushout of two finite mappings using connected components.

    Parameters:
        left : The size of the left set.
        right : The size of the right set.
        left_boundary : The mapping from boundary to left.
        right_boundary : The mapping from boundary to right.

    Examples
    --------
    >>> assert pushout(2, 3, [1], [0]) == ({0: 0, 1: 1}, {0: 1, 1: 2, 2: 3})
    """
    if len(left_boundary) != len(right_boundary):
        raise ValueError
    components, left_pushout, right_pushout = set(), {}, {}
    left_proper = sorted(set(range(left)) - set(left_boundary))
    left_pushout.update({j: i for i, j in enumerate(left_proper)})
    graph = Graph([
        (("middle", i), ("left", j)) for i, j in enumerate(left_boundary)] + [
        (("middle", i), ("right", j)) for i, j in enumerate(right_boundary)])
    for i, component in enumerate(connected_components(graph)):
        components.add(i)
        for case, j in component:
            if case == "left":
                left_pushout[j] = len(left_proper) + i
            if case == "right":
                right_pushout[j] = len(left_proper) + i
    right_proper = set(range(right)) - set(right_boundary)
    right_pushout.update({
        j: len(left_proper) + len(components) + i
        for i, j in enumerate(right_proper)})
    return left_pushout, right_pushout


class BinaryBoxConstructor:
    """
    Box constructor with attributes ``left`` and ``right`` as input.

    Parameters:
        left : Some attribute on the left.
        right : Some attribute on the right.
    """
    def __init__(self, left, right):
        self.left, self.right = left, right

    def __setstate__(self, state):
        if "_name" in state:
            state["_name"] = type(self).__name__ + (
                              f"({state['right']}, {state['left']})"
                              if state.get("_is_dagger", False) else
                              f"({state['left']}, {state['right']})"
            )
        super().__setstate__(state)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({repr(self.left)}, {repr(self.right)})"

    def to_tree(self) -> dict:
        """ Serialise a binary box constructor. """
        left, right = self.left.to_tree(), self.right.to_tree()
        return dict(factory=factory_name(type(self)), left=left, right=right)

    @classmethod
    def from_tree(cls, tree: dict) -> BinaryBoxConstructor:
        """ Decode a serialised binary box constructor. """
        return cls(*map(from_tree, (tree['left'], tree['right'])))


def draw_and_compare(file, folder, tol, **params):
    """ Draw a given diagram and compare the result with a baseline. """
    def decorator(func):
        def wrapper():
            diagram = func()
            draw = params.get('draw', type(diagram).draw)
            true_path = os.path.join(folder, file)
            test_path = os.path.join(folder, '.' + file)
            draw(diagram, path=test_path, show=False, **params)
            test = compare_images(true_path, test_path, tol)
            assert test is None
            os.remove(test_path)
        return wrapper
    return decorator


def tikz_and_compare(file, folder, **params):
    """ Tikz a given diagram and compare the result with a baseline. """
    def decorator(func):
        def wrapper():
            diagram = func()
            draw = params.get('draw', type(diagram).draw)
            true_paths = [os.path.join(folder, file)]
            test_paths = [os.path.join(folder, '.' + file)]
            if params.get("use_tikzstyles", DRAWING_DEFAULT['use_tikzstyles']):
                true_paths.append(
                    true_paths[0].replace('.tikz', '.tikzstyles'))
                test_paths.append(
                    test_paths[0].replace('.tikz', '.tikzstyles'))
            draw(diagram, path=test_paths[0], **dict(params, to_tikz=True))
            for true_path, test_path in zip(true_paths, test_paths):
                with open(true_path, "r") as true:
                    with open(test_path, "r") as test:
                        assert true.read() == test.read()
                os.remove(test_path)
        return wrapper
    return decorator


def tuplify(stuff: any) -> tuple:
    """
    Turns anything into a tuple, do nothing if it is already.

    Parameters:
        stuff : The stuff to turn into a tuple.
    """
    return stuff if isinstance(stuff, tuple) else (stuff, )


def untuplify(stuff: tuple) -> any:
    """
    Takes the element out of a tuple if it has length 1, otherwise do nothing.

    Parameters:
        stuff : The tuple out of which to take the element.

    Important
    ---------
    This is the inverse of :func:`tuplify`, except on tuples of length 1.
    """
    return stuff[0] if len(stuff) == 1 else stuff


T = TypeVar('T')


class Composable(ABC, Generic[T]):
    """
    Abstract class implementing the syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with some method :code:`then`.

    Example
    -------
    >>> class List(list, Composable):
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    factory: Type[Composable]
    sum_factory: Type[Composable]
    ty_factory: Type[T]
    dom: T
    cod: T

    @abstractmethod
    def then(self, other: Optional[Composable[T]], *others: Composable[T]
             ) -> Composable[T]:
        """
        Sequential composition, to be instantiated.

        Parameters:
            other : The other composable object to compose sequentially.
        """

    def is_composable(self, other: Composable) -> bool:
        """
        Whether two objects are composable, i.e. the codomain of the first is
        the domain of the second.

        Parameters:
            other : The other composable object.
        """
        return self.cod == other.dom

    def is_parallel(self, other: Composable) -> bool:
        """
        Whether two composable objects are parallel, i.e. they have the same
        domain and codomain.

        Parameters:
            other : The other composable object.
        """
        return (self.dom, self.cod) == (other.dom, other.cod)

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


def factory(cls: Type[Composable]) -> Type[Composable]:
    """
    Allows the identity and composition of an :class:`Arrow` subclass to remain
    within the subclass.

    Parameters:
        cls : Some subclass of :class:`Arrow`.

    Note
    ----
    The factory method pattern (`FMP`_) is used all over DisCoPy.

    .. _FMP: https://en.wikipedia.org/wiki/Factory_method_pattern

    Example
    -------
    Let's create :code:`Circuit` as a subclass of :class:`Arrow` with an
    :class:`Ob` subclass :code:`Qubit` as domain and codomain.

    >>> from discopy.cat import Ob, Arrow, Box
    >>> class Qubit(Ob):
    ...     pass
    >>> @factory
    ... class Circuit(Arrow):
    ...     ty_factory = Qubit

    The :code:`Circuit` subclass itself has a subclass :code:`Gate` as boxes.

    >>> class Gate(Box, Circuit):
    ...     pass

    The identity and composition of :code:`Circuit` is again a :code:`Circuit`.

    >>> X = Gate('X', Qubit(), Qubit())
    >>> assert isinstance(X >> X, Circuit)
    >>> assert isinstance(Circuit.id(), Circuit)
    >>> assert isinstance(Circuit.id().dom, Qubit)
    """
    cls.factory = cls
    return cls


class Whiskerable(ABC):
    """
    Abstract class implementing the syntactic sugar :code:`@` for whiskering
    and parallel composition with some method :code:`tensor`.
    """
    @classmethod
    @abstractmethod
    def id(cls, dom: any) -> Whiskerable:
        """
        Identity on a given domain, to be instantiated.

        Parameters:
            dom : The object on which to take the identity.
        """

    @abstractmethod
    def tensor(self, other: Whiskerable) -> Whiskerable:
        """
        Parallel composition, to be instantiated.

        Parameters:
            other : The other diagram to compose in parallel.
        """

    @classmethod
    def whisker(cls, other: any) -> Whiskerable:
        """
        Apply :meth:`Whiskerable.id` if :code:`other` is not tensorable else do
        nothing.

        Parameters:
            other : The whiskering object.
        """
        return other if isinstance(other, Whiskerable) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class AxiomError(Exception):
    """ The gods of category theory are not happy. """


def assert_iscomposable(left: Composable, right: Composable):
    """
    Raise :class:`AxiomError` if two objects are not composable,
    i.e. the domain of ``other`` is not the codomain of ``self``.

    Parameters:
        left : A composable object.
        right : Another composable object.
    """
    if not left.is_composable(right):
        raise AxiomError(messages.NOT_COMPOSABLE.format(
            left, right, left.cod, right.dom))


def assert_isparallel(left: Composable, right: Composable):
    """
    Raise :class:`AxiomError` if two composable objects do not have the
    same domain and codomain.

    Parameters:
        left : A composable object.
        right : Another composable object.
    """
    if not left.is_parallel(right):
        raise AxiomError(messages.NOT_PARALLEL.format(left, right))


def assert_isatomic(typ: Ty, cls: type = None):
    """ Raise :class:`AxiomError` if a type does not have length one. """
    cls = cls or type(typ)
    assert_isinstance(typ, cls)
    if not typ.is_atomic:
        raise ValueError(messages.NOT_ATOMIC.format(
            factory_name(cls), len(typ)))


def assert_istraceable(arg: Diagram, n=1, left=False):
    """ Raise :class:`AxiomError` if a diagram is not traceable. """
    traced_dom, traced_cod = (arg.dom[:n], arg.cod[:n]) if left\
        else (arg.dom[len(arg.dom) - n:], arg.cod[len(arg.cod) - n:])
    if traced_dom != traced_cod:
        raise AxiomError(
            messages.NOT_TRACEABLE.format(traced_dom, traced_cod))


class classproperty(object):
    """ Adapted from https://stackoverflow.com/a/5192374/18783670 """
    def __init__(self, f):
        self.f = f

    def __get__(self, _, x):
        return self.f(x)
