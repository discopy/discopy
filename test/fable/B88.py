"""B88: docs, messages and config drift — CONTRIBUTING, CHANGELOG, readme.py, bare raises, and a dozen small API rows (CONTRIBUTING.md:125, CHANGELOG.md:230,322, readme.py, abc.py, cmap.py, hypergraph.py, markov.py:148, cat.py:905, abc.py:184, para.py:380, symmetric.py:187, feedback.py:478, tensor.py:416, python/multiplicative.py:184).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import inspect
import re
import runpy
import time
from pathlib import Path

import pytest

from discopy import abc, feedback, markov, monoidal, para, symmetric, tensor
from discopy.monoidal import Box, Functor, Ty
from discopy.python import multiplicative
from discopy.tensor import Dim

ROOT = Path(__file__).parents[2]
SRC = ROOT / 'discopy'


def test_b88_contributing_names_the_real_build_backend():
    contributing = (ROOT / 'CONTRIBUTING.md').read_text()
    pyproject = (ROOT / 'pyproject.toml').read_text()
    assert ('uv_build' not in contributing
            or 'build-backend = "uv_build"' in pyproject)


def test_b88_changelog_persist_credentials_claim_matches_workflows():
    changelog = (ROOT / 'CHANGELOG.md').read_text()
    workflows = list((ROOT / '.github' / 'workflows').glob('*.yml'))
    checkouts = sum(
        w.read_text().count('actions/checkout@') for w in workflows)
    persisted = sum(
        w.read_text().count('persist-credentials: false') for w in workflows)
    assert ('persist-credentials: false' not in changelog
            or checkouts == persisted)


def test_b88_root_readme_script_runs_or_is_gone():
    script = ROOT / 'readme.py'
    if not script.exists():
        return
    runpy.run_path(str(script), run_name='__main__')


def test_b88_changelog_ty_name_line_matches_ty_name():
    changelog = (ROOT / 'CHANGELOG.md').read_text()
    claim = 'computed from its `inside`'
    assert claim not in changelog or Ty('x').name != Ty('y').name


@pytest.mark.parametrize('module', ['abc', 'cmap', 'hypergraph'])
def test_b88_no_message_less_raise(module):
    src = (SRC / f'{module}.py').read_text()
    bare = re.findall(r'^\s*raise (ValueError|AxiomError)(\(\))?\s*$',
                      src, re.MULTILINE)
    assert len(bare) == 0, f'{len(bare)} message-less raises in {module}.py'


def test_b88_sum_of_a_thousand_terms_builds_in_under_a_second():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    deadline, total = time.perf_counter() + 1, 0
    for _ in range(1000):
        total = total + f
        if time.perf_counter() > deadline:
            pytest.fail(f'one second gone after {len(total.terms)} terms')


def test_b88_markov_discard_takes_n():
    x = markov.Ty('x')
    try:
        d = markov.Diagram.discard(x, 5)
    except TypeError:
        return
    assert d.dom == x ** 5 and d.cod == markov.Ty()


def test_b88_functor_eq_distinguishes_dom():
    F = Functor({}, {}, dom=monoidal.Diagram)
    G = Functor({}, {}, dom=symmetric.Diagram)
    assert F != G


def test_b88_abstract_tensor_kind_matches_the_implementations():
    abstract = inspect.getattr_static(abc.MonoidalCategory, 'tensor')
    concrete = inspect.getattr_static(monoidal.Diagram, 'tensor')
    assert isinstance(abstract, classmethod) == isinstance(
        concrete, classmethod)


def test_b88_para_closed_curry_default_matches_abc():
    def default(method):
        return inspect.signature(method).parameters['left'].default
    assert default(para.Closed.curry) == default(abc.ClosedCategory.curry)


def test_b88_foliation_merges_plumbing_that_cancels():
    X = symmetric.Ty(*'abc')
    p = symmetric.Permutation(X, [1, 2, 0])
    assert len((p >> p[::-1] >> p).foliation().inside) == 1


@pytest.mark.parametrize('box', [
    lambda x: feedback.Copy(x),
    lambda x: feedback.Merge(x),
    lambda x: feedback.Permutation(x @ x, [1, 0])], ids=[
    'Copy', 'Merge', 'Permutation'])
def test_b88_feedback_plumbing_delay_sets_time_step(box):
    assert box(feedback.Ty('x')).delay(42).time_step == 42


def test_b88_tensor_functor_maps_dim():
    F = tensor.Functor({Dim(2): Dim(3)}, {}, dom=tensor.Diagram)
    assert F(Dim(2)) == Dim(3)


def test_b88_multiplicative_curry_too_many_raises():
    f = multiplicative.Function(lambda a: a, (int, ), (int, ))
    with pytest.raises((ValueError, IndexError)):
        f.curry(5)
