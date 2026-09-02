"""B83: COLORS is indexed bare on spiders, ribbons and every TikZ path, and TikZ gets `->looseness` and `scale=12` (discopy/drawing/backend.py:840, 978, 1079, 1181, 1337).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import re

import matplotlib

from discopy import ribbon
from discopy.monoidal import Box, Ty
from discopy.quantum import Measure

matplotlib.use('Agg')

x = Ty('x')
X = ribbon.Ty('x')


def test_b83_lightgrey_box_draws_control(tmp_path):
    Box('f', x, x, color='lightgrey').draw(
        path=str(tmp_path / 'box.svg'), show=False)


def test_b83_lightgrey_spider_draws(tmp_path):
    Box('f', x, x, draw_as_spider=True, color='lightgrey').draw(
        path=str(tmp_path / 'spider.svg'), show=False)


def test_b83_lightgrey_ribbon_draws_to_tikz(tmp_path):
    ribbon.Diagram.cups(X, X.r).to_ribbons(colour=lambda ob: 'lightgrey').draw(
        path=str(tmp_path / 'cup.tikz'), to_tikz=True)


def test_b83_lightgrey_twist_draws(tmp_path):
    ribbon.Diagram.twist(X).to_ribbons(colour=lambda ob: 'lightgrey').draw(
        path=str(tmp_path / 'twist.svg'), show=False)


def test_b83_measure_tikz_separates_style_keys(tmp_path):
    path = tmp_path / 'measure.tikz'
    Measure().draw(path=str(path), to_tikz=True)
    text = path.read_text()
    assert '->looseness' not in text
    for options in re.findall(r'\\draw \[([^\]]*)\]', text):
        for key in options.split(', '):
            assert re.fullmatch(r'[-<>]+|[a-z ]+(=[^=]*)?', key), options


def test_b83_fontsize_is_not_a_tikz_scale_factor(tmp_path):
    path = tmp_path / 'box.tikz'
    Box('f', x, x).draw(path=str(path), to_tikz=True, fontsize=12)
    assert 'scale=12' not in path.read_text()
