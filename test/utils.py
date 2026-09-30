from os import listdir

import pickle
import re

import pytest
from pytest import warns

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import cat, rigid
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


def test_encode_complex():
    assert encode_complex(1 + 2j) == {
        'factory': 'complex', 're': 1.0, 'im': 2.0}
    assert encode_complex(complex(0)) == {
        'factory': 'complex', 're': 0.0, 'im': 0.0}
    with pytest.raises(TypeError) as err:
        encode_complex(Ob('x'))
    assert "Ob" in str(err.value)


def test_decode_complex():
    assert decode_complex(encode_complex(1 + 2j)) == 1 + 2j
    for tree in ({'factory': 'cat.Ob', 'name': 'x'},
                 {'re': 1, 'im': 2},
                 {'factory': 'complex', 're': 1},
                 {'factory': 'complex', 're': 1, 'im': 2, 'name': 'x'}):
        assert decode_complex(tree) == tree


@pytest.mark.parametrize('data', [
    1j, [1, 1j], [[1j, 2], {'key': 3j}], {'re': 'a', 'im': 'b'},
    {'factory': 'cat.Ob', 'name': 'x'}])
def test_dumps_complex_data(data):
    """ JSON has no complex numbers, so ``dumps`` encodes them as a pair of
    reals and ``loads`` reads them back, leaving every other datum alone. """
    box = cat.Box('f', Ob('x'), Ob('y'), data=data)
    assert loads(dumps(box)) == box


def test_dumps_complex_gate():
    """ Every quantum gate carries a complex array, so none of them used to
    serialise at all (#775). """
    from discopy.quantum.gates import H, X, Y, Z, S, T
    for gate in (H, X, Y, Z, S, T, H >> X, H @ X):
        assert loads(dumps(gate)) == gate
