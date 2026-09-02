"""B84: post.py dying before record() leaves `clean` unset, and benchmark_comment.mismatch crashes on a deleted fork (.github/style-review/post.py:245, .github/scripts/benchmark_comment.py:100).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

GITHUB = Path(__file__).parents[2] / '.github'
STYLE_REVIEW = GITHUB / 'style-review'

if str(STYLE_REVIEW) not in sys.path:
    sys.path.insert(0, str(STYLE_REVIEW))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b84_post_records_clean_false_on_unreadable_findings(
        tmp_path, monkeypatch):
    history, post = load(STYLE_REVIEW / 'history.py'), load(
        STYLE_REVIEW / 'post.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'output'))
    monkeypatch.setenv('MODEL', 'a-model')
    monkeypatch.setattr(post, 'moved', lambda: None)
    os.makedirs(history.DIRECTORY)
    with open(os.path.join(history.DIRECTORY, 'history.json'), 'w') as file:
        json.dump(history.empty(), file)
    with open(os.path.join(history.DIRECTORY, 'findings.json'), 'w') as file:
        json.dump({'findings': [
            {'path': 'a.py', 'line': 'middle', 'comment': 'x'}],
            'coverage': {}}, file)
    (tmp_path / history.DIRECTORY / 'diff.patch').write_text('')
    with contextlib.suppress(ValueError):
        post.main()
    assert (tmp_path / 'output').exists(), "GITHUB_OUTPUT never written"
    assert 'clean=false' in (tmp_path / 'output').read_text()


def test_b84_mismatch_refuses_a_deleted_fork():
    benchmark_comment = load(GITHUB / 'scripts' / 'benchmark_comment.py')
    run = {'head_repository': {'full_name': 'x/y'}, 'head_branch': 'b'}
    pull = {'base': {'repo': {'full_name': 'discopy/discopy'}},
            'head': {'repo': None, 'ref': 'b'}}
    reason = benchmark_comment.mismatch({}, run, pull, 'discopy/discopy')
    assert isinstance(reason, str) and reason
