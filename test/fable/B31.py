"""B31: pregroup normal_form crashes on foliated diagrams and left-whiskered words (discopy/grammar/pregroup.py:110).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.grammar.pregroup import Ty, Word, Cup, Id

s, n = Ty('s'), Ty('n')


def sentence():
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ s @ Cup(n.l, n)


def test_b31_foliation_normal_form():
    assert sentence().foliation().normal_form() == sentence().normal_form()


def test_b31_left_whiskered_word_normal_form():
    diagram = Id(n) @ Word('Alice', n)
    assert diagram.normal_form().cod == n @ n
