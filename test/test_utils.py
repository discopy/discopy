import pytest

from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import Ob
from discopy.quantum.gates import Rx
from discopy.quantum.zx import H as HBox, Z as ZSpider, scalar
from discopy.utils import *

gates = [
    Rx(1), HBox, ZSpider(2, 4), scalar(1),
]

zip_mock = MagicMock()
zip_mock.open().__enter__().read.return_value =\
    '[{"factory": "discopy.cat.Ob", "name": "a"}]'


@patch('urllib.request.urlretrieve', return_value=(None, None))
@patch('zipfile.ZipFile', return_value=zip_mock)
def test_load_corpus(a, b):
    assert load_corpus("[fake url]") == [Ob("a")]


@pytest.mark.parametrize('c', gates)
def test_pickled_gate_serialisation(c):
    assert c == loads(dumps(c))
