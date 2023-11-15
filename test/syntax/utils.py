from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import rigid
from discopy.cat import Ob
from discopy.utils import *

import pytest
from pytest import warns

from os import listdir

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


# @pytest.mark.parametrize('fn', listdir('test/src/pickles/main/'))
def test_pickle(fn = "quantum.Circuit.pickle"):
    import pickle
    with open(f"test/src/pickles/main/{fn}", 'rb') as f:
        new = pickle.load(f)
    with open(f"test/src/pickles/0.6/{fn}", 'rb') as f:
        old = pickle.load(f)
    if not hasattr(new, "__dict__"):
        return old == new
    old_d, new_d = old.__dict__, new.__dict__
    cod = old_d["cod"] == new_d["cod"]
    dom = old_d["dom"] == new_d["dom"]
    d1, d2 = tuple(old_d["inside"][5])[1], tuple(new_d["inside"][5])[1]
    d1 == d2
    for i, (a, b) in enumerate(zip(old_d["inside"], new_d["inside"])):
        x = a == b
        if not x:
            a == b
    assert old_d == new_d and all(
        old_d[key].__dict__ == new_d[key].__dict__
        for key in list(old_d.keys()) if hasattr(old_d[key], "__dict__"))
