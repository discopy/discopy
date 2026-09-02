"""Load the scripts under test by path, since ``.github`` is not a package.

They are side-effect free at import: each does its work under
``if __name__ == "__main__"`` and reads the environment inside functions.
The style reviewer's scripts import one another by bare name, as they do
on a runner where the interpreter is given one of them, so their
directory goes on the path before any of them is loaded. The tests are
named after the script each covers, which is the same name again, so they
are collected with ``--import-mode=importlib``: it leaves both the path
and ``sys.modules`` alone, where the default mode would import a test
file as the module its subject imports.
"""

import importlib.util
import pathlib
import sys

import pytest

GITHUB = pathlib.Path(__file__).resolve().parent.parent
STYLE_REVIEW = GITHUB / "style-review"

if str(STYLE_REVIEW) not in sys.path:
    sys.path.insert(0, str(STYLE_REVIEW))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def review():
    return load(STYLE_REVIEW / "review.py")


@pytest.fixture(scope="session")
def post():
    return load(STYLE_REVIEW / "post.py")


@pytest.fixture(scope="session")
def history():
    return load(STYLE_REVIEW / "history.py")


@pytest.fixture(scope="session")
def thread():
    return load(STYLE_REVIEW / "thread.py")


@pytest.fixture(scope="session")
def benchmark_comment():
    return load(GITHUB / "scripts" / "benchmark_comment.py")
