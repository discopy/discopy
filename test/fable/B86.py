"""B86: transparency beyond B39 — reprs and trees dropping attributes, unevaluable reprs, a permutation that is not a Swap, no CMap.to_tree, unhashable dataclasses (discopy/cat.py:788, biclosed.py:355, grammar/pregroup.py:204, grammar/cfg.py:82, feedback.py:606, symmetric.py:465, cmap.py, para.py:156, stream.py:299, interaction.py:156).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy import (cat, monoidal, symmetric, rigid, compact, biclosed,
                     feedback, para, stream, interaction, ribbon, grammar)
from discopy.grammar import cfg, pregroup
from discopy.utils import dumps, loads

NS = dict(cat=cat, monoidal=monoidal, symmetric=symmetric, rigid=rigid,
          biclosed=biclosed, feedback=feedback, grammar=grammar)


def test_b86_cat_bubble_repr_keeps_name_and_method():
    bubble = cat.Box('f', cat.Ob('x'), cat.Ob('y')).bubble(name="N", method="m")
    assert eval(repr(bubble), dict(NS)) == bubble


def test_b86_cat_bubble_tree_keeps_name_and_method():
    bubble = cat.Box('f', cat.Ob('x'), cat.Ob('y')).bubble(name="N", method="m")
    assert loads(dumps(bubble)) == bubble


x, y = biclosed.Ty('x'), biclosed.Ty('y')


def test_b86_eval_and_coeval_repr():
    for box in (biclosed.Eval(x << y), biclosed.Coeval(x << y)):
        assert eval(repr(box), dict(NS)) == box


def test_b86_eval_and_coeval_tree():
    for box in (biclosed.Eval(x << y), biclosed.Coeval(x << y)):
        assert loads(dumps(box)) == box


def test_b86_rotated_word_tree_keeps_z():
    word = pregroup.Word('Alice', pregroup.Ty('n')).r
    loaded = loads(dumps(word))
    assert loaded.z == 1 and loaded == word


def test_b86_word_repr_carries_the_factory_prefix():
    word = pregroup.Word('Alice', pregroup.Ty('n'))
    assert eval(repr(word), dict(NS)) == word


n = cfg.Ty('n')
rule, leaf = cfg.Rule(n @ n, n, name='r'), cfg.Word('a', n)


def test_b86_cfg_reprs_are_evaluable():
    for tree in (rule, rule(leaf, leaf), cfg.Id(n)):
        assert eval(repr(tree), dict(NS)) == tree


def test_b86_cfg_rule_equals_its_trivial_tree():
    assert cfg.Tree(leaf) == leaf and leaf == cfg.Tree(leaf)


def test_b86_followed_by_repr_carries_the_module_prefix():
    box = feedback.FollowedBy(feedback.Ty('x'))
    assert eval(repr(box), dict(NS)) == box


def test_b86_permutation_from_iterator_is_a_swap():
    sx, sy = symmetric.Ty('x'), symmetric.Ty('y')
    perm = symmetric.Permutation(sx @ sy, iter([1, 0]))
    assert isinstance(perm, symmetric.Swap) and perm == symmetric.Swap(sx, sy)


def test_b86_cmap_tree():
    cx, cy = compact.Ty('x'), compact.Ty('y')
    cmap = compact.CMap.from_box(compact.Box('f', cx, cy))
    assert loads(dumps(cmap)) == cmap


def test_b86_para_is_hashable():
    sx, sy = symmetric.Ty('x'), symmetric.Ty('y')
    box = symmetric.Box('f', sx, sy)
    assert hash(para.Symmetric[symmetric.Diagram](sx, sy, box)) is not None


def test_b86_stream_is_hashable():
    sx = stream.Ty('x')
    assert hash(stream.Stream.sequence('f', sx, sx)) is not None


def test_b86_interaction_diagram_is_hashable():
    x0, x1 = map(ribbon.Ty, ["x0", "x1"])
    X = interaction.Ty[ribbon.Ty](x0, x1)
    box = ribbon.Box('f', x0 @ x1, x0 @ x1)
    assert hash(interaction.Diagram[ribbon.Diagram](box, X, X)) is not None
