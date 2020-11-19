from discopy import *
from discopy.grammar.parser import *


def test_parser():
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word("Alice", n), Word("Bob", n)
    loves = Word("loves", n.r @ s @ n.l)
    assert get_parsing([Alice, loves, Bob])\
        == (True, {(1, 0): (0, 0), (2, 0): (1, 2)})
