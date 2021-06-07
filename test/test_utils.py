from unittest.mock import MagicMock
from unittest.mock import patch

from discopy import Ob
from discopy.utils import *


zip_mock = MagicMock()
zip_mock.open().__enter__().read.return_value =\
    '[{"factory": "discopy.cat.Ob", "name": "a"}]'


@patch('urllib.request.urlretrieve', return_value=(None, None))
@patch('zipfile.ZipFile', return_value=zip_mock)
def test_load_corpus(a, b):
    assert load_corpus("[fake url]") == [Ob("a")]
