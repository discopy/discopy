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


def composable_triple(cls):
    """ Three composable morphisms of a concrete category. """
    from discopy import cat, matrix, monoidal
    from discopy.hypergraph import Hypergraph
    from discopy.python import finset, function
    from discopy.quantum.channel import C, Channel
    from discopy.tensor import Dim, Tensor

    x, y = cat.Ob('x'), monoidal.Ty('y')
    return {
        cat.Arrow: lambda: [cat.Box(name, x, x) for name in "fgh"],
        monoidal.Diagram: lambda: [monoidal.Box(n, y, y) for n in "fgh"],
        matrix.Matrix: lambda: 3 * [matrix.Matrix([0, 1, 1, 0], 2, 2)],
        Tensor: lambda: [
            Tensor([0, 1, 1, 0, 1, 0, 0, 1], Dim(2), Dim(2, 2)),
            Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2, 2), Dim(2)),
            Tensor([0, 1, 1, 0], Dim(2), Dim(2))],
        Channel: lambda: 3 * [
            Channel(Tensor.id(Dim(1)).array, C(Dim(1)), C(Dim(1)))],
        function.Function: lambda: 3 * [
            function.Function(lambda n: n + 1, (int, ), (int, ))],
        finset.Function: lambda: 3 * [finset.Function([1, 0], 2, 2)],
        finset.Permutation: lambda: 3 * [finset.Permutation([1, 2, 0])],
        cat.Functor: lambda: 3 * [cat.Functor({x: x}, {})],
        cat.Transformation: lambda: 3 * [
            cat.Transformation.id(cat.Functor({x: x}, {}))],
        monoidal.Functor: lambda: 3 * [monoidal.Functor({y: y}, {})],
        Hypergraph: lambda: 3 * [monoidal.Box('f', y, y).to_hypergraph()],
    }[cls]()


def implementors():
    """ Every concrete class implementing ``abc.Category.then``. """
    from discopy import cat, matrix, monoidal
    from discopy.hypergraph import Hypergraph
    from discopy.python import finset, function
    from discopy.quantum.channel import Channel
    from discopy.tensor import Tensor

    return [
        cat.Arrow, monoidal.Diagram, matrix.Matrix, Tensor, Channel,
        function.Function, finset.Function, finset.Permutation,
        cat.Functor, cat.Transformation, monoidal.Functor, Hypergraph]


def same(f, g):
    """
    Whether two morphisms are equal, or act the same when they cannot be:
    ``python.Function`` and ``cat.Transformation`` are given by closures, so
    two composites that agree everywhere are still never equal.
    """
    from discopy import cat
    from discopy.python import function

    if isinstance(f, function.Function):
        return (f.dom, f.cod) == (g.dom, g.cod) and f(0) == g(0)
    if isinstance(f, cat.Transformation):
        probe = cat.Ob('x')
        return (f.dom, f.cod) == (g.dom, g.cod) and f(probe) == g(probe)
    return f == g


@pytest.mark.parametrize("cls", implementors(), ids=factory_name)
def test_then_is_unbiased(cls):
    """
    ``then`` composes ``n >= 0`` morphisms in every category, as
    :meth:`discopy.abc.Category.then` declares.

    Python does not check an override's signature, so a category that
    composes exactly two morphisms satisfies the abstract method while
    breaking its contract: ``Tensor.then(g, h)`` used to raise
    :class:`ValueError`, and every binary ``then`` used to raise on no
    argument at all.
    """
    f, g, h = composable_triple(cls)
    assert same(f.then(), f)
    assert same(f.then(g), f >> g)
    assert same(f.then(g, h), (f >> g) >> h)


@pytest.mark.parametrize("cls", implementors(), ids=factory_name)
def test_then_signature(cls):
    """
    Every ``then`` advertises the unbiased signature it implements, whether
    it is n-ary itself or wrapped in :func:`discopy.utils.unbiased`.
    """
    from inspect import Parameter, signature

    parameters = list(signature(cls.then).parameters.values())
    assert parameters[1].kind == Parameter.VAR_POSITIONAL


def test_then_tensor_contracts():
    """
    ``Tensor.then`` contracts every step, where sending ``n >= 3`` to
    ``Matrix.then`` used to multiply the arrays as matrices: for most
    boundaries that raised, but for these three it returned the transpose
    of the right answer, silently.
    """
    from discopy.tensor import Dim, Tensor

    f = Tensor([1, 0, 0, 1, 0, 0], Dim(2), Dim(3))
    g = Tensor([1, 2, 0, 1, 1, 0], Dim(3), Dim(2))
    h = Tensor([1, 0, 0, 1, 0, 1, 1, 0], Dim(2), Dim(2, 2))
    assert f.then(g, h) == (f >> g) >> h
    assert f.then(g, h).array.tolist()\
        == [[[1, 2], [2, 1]], [[1, 2], [2, 1]]]


def test_then_not_composable():
    """ The n-ary composite type checks every step, not just the first. """
    from discopy import cat
    from discopy.tensor import Dim, Tensor

    x, y = cat.Ob('x'), cat.Ob('y')
    f, g = cat.Box('f', x, x), cat.Box('g', y, y)
    with pytest.raises(AxiomError):
        f.then(f, g)
    with pytest.raises(AxiomError):
        Tensor.id(Dim(2)).then(Tensor.id(Dim(2)), Tensor.id(Dim(3)))
