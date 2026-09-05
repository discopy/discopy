"""Load the scripts under test by path, since ``.github`` is not a package.

They are side-effect free at import: each does its work under
``if __name__ == "__main__"`` and reads the environment inside functions,
so loading one by path runs nothing. The tests are collected with
``--import-mode=importlib`` so a test file named after its subject does not
shadow the module it loads.
"""

import importlib.util
import pathlib

import pytest

GITHUB = pathlib.Path(__file__).resolve().parent.parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def benchmark_comment():
    return load(GITHUB / "scripts" / "benchmark_comment.py")
