# -*- coding: utf-8 -*-
"""B67: grammar Word/Rule daggers are mistyped or crash, fc/bc curry one atom, from_nltk drops children (discopy/grammar/thue.py:69, cfg.py:141, categorial.py:103).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import monoidal
from discopy.grammar import categorial, cfg, thue

n, s = monoidal.Ty('n'), monoidal.Ty('s')


def test_b67_thue_word_dagger_swaps_dom_and_cod():
    word = thue.Word('a', n)
    assert (word.dagger().dom, word.dagger().cod) == (n, monoidal.Ty())
    assert (word >> word.dagger()).cod == monoidal.Ty()


def test_b67_categorial_word_dagger_composes():
    Bob = categorial.Word('Bob', categorial.Ty('n'))
    assert (Bob >> Bob.dagger()).cod == categorial.Ty()


def test_b67_thue_rule_dagger_builds():
    rule = thue.Rule(n @ n, s, name='r')
    assert (rule.dagger().dom, rule.dagger().cod) == (s, n @ n)


def test_b67_cfg_word_dagger_builds():
    word = cfg.Word('a', n)
    assert (word.dagger().dom, word.dagger().cod) == (n, monoidal.Ty())


def test_b67_cfg_rule_dagger_builds():
    rule = cfg.Rule(n @ n, s, name='r')
    assert (rule.dagger().dom, rule.dagger().cod) == (s, n @ n)


def test_b67_fc_non_atomic_right_type():
    x, y, z, w = map(categorial.Ty, "xyzw")
    d = categorial.Diagram.fc(x, y, z @ w)
    assert (d.dom, d.cod) == ((x << y) @ (y << (z @ w)), x << (z @ w))


def test_b67_bc_non_atomic_left_type():
    x, y, z, w = map(categorial.Ty, "xyzw")
    d = categorial.Diagram.bc(x @ w, y, z)
    assert (d.dom, d.cod) == (((x @ w) >> y) @ (y >> z), (x @ w) >> z)


def test_b67_from_nltk_keeps_subtree_after_a_word():
    nltk = pytest.importorskip("nltk")
    tree = cfg.Tree.from_nltk(nltk.Tree.fromstring("(S (NP I) (VP saw (NP him)))"))
    assert str(tree) == "S(I, VP(saw, him))"


def test_b67_from_nltk_keeps_every_word():
    nltk = pytest.importorskip("nltk")
    tree = cfg.Tree.from_nltk(nltk.Tree.fromstring("(NP the big dog)"))
    assert str(tree) == "NP(the, big, dog)"
