"""B43: nine modules have no mirrored test file and test_para.py breaks the naming convention (test/).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from pathlib import Path

TEST = Path(__file__).parents[1]

MODULES = ['abc', 'config', 'messages', 'grammar/cfg', 'grammar/dependency',
           'grammar/thue', 'python/function', 'quantum/gates',
           'quantum/pennylane']


def test_b43_every_module_has_a_test_file():
    missing = [m for m in MODULES if not (TEST / f"{m}.py").exists()]
    assert missing == []


def test_b43_para_test_file_mirrors_its_module():
    assert (TEST / 'para.py').exists()
