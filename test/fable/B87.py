"""B87: the test suite skips, leaks, cannot fail or asserts the bug (test/quantum/circuit.py:5, test/drawing/drawing.py:486, test/cmap.py:206, test/python/multiplicative.py:28, test/quantum/tk.py:67).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import re
from pathlib import Path

TEST = Path(__file__).parents[1]


def body(src, name):
    return src.split(f'def {name}(')[1].split('\ndef ')[0]


def test_b87_circuit_tests_are_not_skipped_whole_without_torch():
    header = (TEST / 'quantum' / 'circuit.py').read_text().split('\ndef ')[0]
    assert '"torch"' not in header and "'torch'" not in header
    assert 'import torch' not in header


def test_b87_bell_state_test_restores_the_global_h():
    src = (TEST / 'drawing' / 'drawing.py').read_text()
    text = body(src, 'test_tikz_bell_state')
    assert ('H.draw_as_spider' not in text
            or 'monkeypatch' in text or 'finally' in text)


def test_b87_cmap_test_does_not_assert_the_truth_of_a_map():
    src = (TEST / 'cmap.py').read_text()
    assert not re.search(r'^\s*assert [^=<>!\n]*\.to_map\(\)\.trace\(\)\s*$',
                         src, re.MULTILINE)


def test_b87_grad_and_sum_tests_do_not_compare_int_zero_to_zero():
    src = (TEST / 'quantum' / 'circuit.py').read_text()
    assert not re.search(r'\.grad\(phi\)\.eval\(\) == 0\b', src)
    assert 'assert not Sum([], qubit, qubit).eval()' not in src


def test_b87_multiplicative_test_trace_calls_the_trace():
    src = (TEST / 'python' / 'multiplicative.py').read_text()
    lines = body(src, 'test_trace').splitlines()
    calls = [line for i, line in enumerate(lines) if '.trace(' in line
             and 'raises(NotImplementedError)' not in lines[i - 1]]
    assert calls


def test_b87_tk_test_does_not_assert_the_dropped_swap():
    src = (TEST / 'quantum' / 'tk.py').read_text()
    flat = ' '.join(src.replace('\\\n', ' ').split())
    assert 'Id(qubit @ bit).init_and_discard() == back_n_forth(Swap(' \
        not in flat
