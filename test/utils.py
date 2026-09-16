from os import listdir

import pickle
import re

import pytest
from pytest import warns

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import rigid
from discopy.cat import Ob
from discopy.utils import *
from discopy.tensor import Box

import pytest
from pytest import warns

from os import listdir
import pickle

zip_mock = MagicMock()
zip_mock.open().__enter__().read.return_value =\
    '[{"factory": "cat.Ob", "name": "a"}]'


@patch('urllib.request.urlretrieve', return_value=(None, None))
@patch('zipfile.ZipFile', return_value=zip_mock)
def test_load_corpus(a, b):
    assert load_corpus("[fake url]") == [Ob("a")]


def test_deprecated_from_tree():
    tree = {
        'factory': 'discopy.rigid.Diagram',
        'dom': {'factory': 'discopy.rigid.Ty',
                'objects': [{'factory': 'discopy.rigid.Ob', 'name': 'n'}]},
        'cod': {'factory': 'discopy.rigid.Ty',
                'objects': [{'factory': 'discopy.rigid.Ob', 'name': 'n'}]},
        'boxes': [], 'offsets': []}
    with warns(DeprecationWarning):
        assert from_tree(tree) == rigid.Id(rigid.Ty('n'))


def test_named_generic_cache():
    from discopy import tensor as dt
    box, box_int, box_float = dt.Box, dt.Box[int], dt.Box[float]
    assert box_int is dt.Box[int]
    assert box is not box_int and box_float is not box_int
    diag_int = dt.Diagram[int]
    assert diag_int is dt.Diagram[int]
    assert box_int is dt.Box[int]



def _rounded_repr(obj):
    # Gate matrices such as the Hadamard's 1 / sqrt(2) entries are stored as
    # floats whose last bit depends on the numpy version that generated the
    # pickle, so exact equality across versions is not portable. Round every
    # float in the repr to 12 significant figures before comparing.
    return re.sub(
        r'\d+\.\d+', lambda m: format(float(m.group()), '.12g'), repr(obj))


@pytest.mark.parametrize('version', ['0.6', '1.2'])
@pytest.mark.parametrize('fn', listdir('test/fixtures/pickles/1.3/'))
def test_pickle_version_compatibility(fn, version):
    if fn == 'quantum.Circuit.pickle':
        pytest.importorskip("pytket")
    with open(f"test/fixtures/pickles/1.3/{fn}", 'rb') as f:
        new = pickle.load(f)
    with open(f"test/fixtures/pickles/{version}/{fn}", 'rb') as f:
        old = pickle.load(f)
    assert old == new or _rounded_repr(old) == _rounded_repr(new)


def test_parameterised_box_pickle():
    box = Box("A", 2, 3)
    assert pickle.loads(pickle.dumps(box)) == box


def test_deprecated_ob():
    from discopy import (
        biclosed, braided, compact, feedback, frobenius, pivotal, rigid)
    from discopy.grammar import pregroup
    from discopy.quantum import circuit
    for module in (rigid, braided, biclosed, pivotal, frobenius, feedback,
                   circuit, pregroup, compact):
        with warns(DeprecationWarning):
            assert module.Ob is module.Wire
        with pytest.raises(AttributeError):
            module.not_an_attribute


def test_wire_tree_roundtrip():
    from discopy import biclosed, braided, feedback, frobenius, pivotal, rigid
    from discopy.quantum import circuit
    for x in (rigid.Wire('x'), braided.Wire('x'), biclosed.Wire('x'),
              pivotal.Wire('x'), frobenius.Wire('x'), feedback.Wire('x'),
              circuit.Digit(2)):
        assert from_tree(x.to_tree()) == x
    with warns(DeprecationWarning):
        assert from_tree({'factory': 'discopy.frobenius.Ob', 'name': 'x'})\
            == frobenius.Wire('x')


def test_factory_roots():
    from discopy import cat, symmetric, closed
    assert cat.Arrow.generator_factory is cat.Box
    assert symmetric.Diagram.swap_factory is symmetric.Swap
    assert symmetric.Diagram.braid_factory is symmetric.Swap
    assert closed.Diagram.braid_factory is closed.Swap
    assert closed.Swap.swap_factory is closed.Swap


def test_factory_bases():
    from discopy import markov, closed, symmetric, ribbon, compact, frobenius
    from discopy import biclosed, tensor
    assert closed.Swap.__bases__ == (markov.Swap, closed.Permutation)
    assert closed.Discard.__bases__ == (markov.Discard, closed.Copy)
    assert closed.Sum.__bases__ == (markov.Sum, biclosed.Sum, closed.Box)
    assert compact.Swap.__bases__ == (symmetric.Swap, compact.Permutation)
    assert not issubclass(compact.Swap, ribbon.Braid)
    assert frobenius.Permutation.__bases__ == (
        compact.Permutation, markov.Permutation, frobenius.Box)
    assert issubclass(tensor.Swap, tensor.Permutation)
    assert closed.Swap.__module__ == "discopy.closed"
    assert closed.Swap.__name__ == closed.Swap.__qualname__ == "Swap"
    assert factory_name(closed.Swap) == "closed.Swap"


def test_factory_override():
    from discopy import symmetric, tensor

    @factory
    class Recipe(symmetric.Diagram):
        pass

    class Step(symmetric.Box, Recipe):
        pass

    Recipe.generator_factory = Step
    assert Recipe.generator_factory is Step
    assert Recipe.swap_factory.__bases__ == (
        symmetric.Swap, Recipe.permutation_factory)
    assert Recipe.permutation_factory.__bases__ == (
        symmetric.Permutation, Step)
    assert Recipe.swap_factory is Recipe.swap_factory

    class Undecorated(Recipe):
        pass

    assert Undecorated.swap_factory is Recipe.swap_factory
    assert tensor.Diagram[complex].swap_factory is tensor.Swap


def test_factory_level_box():
    """ A root initialises as a box of the level it is built in. """
    from discopy import compact, feedback
    x = compact.Ty('x')
    assert compact.Swap(x, x).z == 0
    assert compact.Swap(x, x).r == compact.Swap(x.r, x.r)
    y = feedback.Ty('y')
    assert feedback.Copy(y).delay().dom == y.delay()
    assert feedback.Swap(y, y).delay().dom == y.delay() @ y.delay()


MODULES = [
    "braided", "traced", "balanced", "symmetric", "markov", "closed",
    "biclosed", "rigid", "pivotal", "ribbon", "compact", "frobenius",
    "feedback", "tensor", "grammar.pregroup", "grammar.categorial"]


def structure(module):
    """ Every generator a level builds through its public methods. """
    from importlib import import_module
    from discopy import abc, monoidal
    module = import_module(f"discopy.{module}")
    D = module.Diagram
    x, y = (D.ob(2), D.ob(3)) if issubclass(D.ob, monoidal.Dim)\
        else (D.ob('x'), D.ob('y'))
    f = module.Box('f', x @ x, x @ x)
    terms = [f.bubble(), f + f]
    if issubclass(D, abc.TracedCategory):
        terms.append(f.trace())
    if issubclass(D, abc.BraidedCategory):
        terms.append(D.braid(x, y))
    if issubclass(D, abc.BalancedCategory):
        terms.append(D.twist(x))
    if issubclass(D, abc.SymmetricCategory):
        terms += [D.swap(x, y), D.permutation([1, 0], [x, y])]
    if issubclass(D, abc.MarkovCategory):
        terms += [D.copy(x), D.merge(x), D.discard(x)]
    if issubclass(D, abc.BiclosedCategory):
        terms += [f.curry(), D.ev(x, y)]
    if issubclass(D, abc.RigidCategory):
        terms += [D.cups(x, x.r), D.caps(x.r, x)]
    if issubclass(D, abc.HypergraphCategory):
        terms.append(D.spiders(1, 2, x))
    return D, terms


@pytest.mark.parametrize("module", MODULES)
def test_factory_exports(module):
    """
    Every generator a level builds is a diagram of that level, defined in
    its module under its own name, so that its representation and its
    pickle can find it.
    """
    import sys
    D, terms = structure(module)
    for term in terms:
        for cls in {type(term)} | {type(box) for box in term.boxes}:
            assert issubclass(cls, D)
            assert getattr(sys.modules[cls.__module__], cls.__name__) is cls
            assert pickle.loads(pickle.dumps(cls)) is cls


@pytest.mark.parametrize("module", MODULES)
def test_factory_extends_bases(module):
    """ Every generator of a level extends those of each of its bases. """
    D, _ = structure(module)
    definitions = {}
    for klass in reversed(D.__mro__):
        definitions.update(vars(klass))
    names = {name for name, value in definitions.items()
             if isinstance(value, (type, cached_classproperty))
             and isinstance(getattr(D, name), type)
             and issubclass(getattr(D, name), D)}
    assert names and all(
        issubclass(getattr(D, name), root)
        for name in names for base in D.__bases__
        if isinstance(root := getattr(base, name, None), type))
