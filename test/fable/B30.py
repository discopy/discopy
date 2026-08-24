"""B30: grammar/cfg.py is broken at every entry point and dependency.py leaves never become Words (discopy/grammar/cfg.py:110, discopy/grammar/dependency.py:40).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.grammar import cfg, dependency
from discopy.monoidal import Ty

x = Ty('x')


def test_b30_rule_hashable():
    f = cfg.Rule(x @ x, x, name='f')
    assert hash(f) == hash(cfg.Rule(x @ x, x, name='f'))
    assert {f: f}[f] == f


def test_b30_tree_builds_and_compares():
    f, w = cfg.Rule(x @ x, x, name='f'), cfg.Word('w', x)
    tree = f(f(w, w), cfg.Id(x))
    assert tree.cod == x
    assert (f(w, w) == 5) is False


def test_b30_word_takes_drawing_params():
    assert cfg.Word('w', x, draw_as_spider=True).name == 'w'


class Token:
    """A stub spaCy token: ``children`` is a generator, like spaCy's."""
    text, dep_ = 'word', 'ROOT'

    @property
    def children(self):
        return iter(())


def test_b30_dependency_leaf_is_word():
    assert isinstance(dependency.doc2tree(Token()), cfg.Word)
