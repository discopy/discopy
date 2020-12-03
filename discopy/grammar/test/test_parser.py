from discopy import *
from discopy.grammar import *
from discopy.grammar.parser import is_sentence, LinPP

parser = LinPP(bound=7)


#dictionary
s, n = Ty('s'), Ty('n')

# Names
Alice = Word('Alice', n)
Bob = Word('Bob', n)

# English
loves = Word('loves', n.r @ s @ n.l)
who = Word('who', n.r @ n @ s.l @ n )
Is = Word('is', n.r @ s @ n.l)
rich = Word('rich', n)
whom = Word('whom', n.r @ n @ s.l @ n.l.l)
about = Word('about', s.r @ n.l @ s)
Chris = Word('Chris', n)
told = Word('told', n.r @ s @ n.l)
you = Word('you', n)


 # Italian
cavalcava = Word('cavalcava', n.r @ s @ n.l)  # 'was riding'
cavallo = Word('cavallo', n)  # 'horse'
bianco = Word('bianco', n.r @ n)  # 'white'(Italian adjectives come after the nouns)
che = Word('che', n.r @ n @ s.l @ n)  # 'who'
nitriva = Word('nitriva', n.r @ s) # 'was neighing'

# Extra
A = Word('A', n.l @ s.l @ n)
B = Word('B', n.r @ s @ n @ n.l)
C = Word('C', n)
D = Word('D', n.r)
E = Word('E', s.r @ s @ n)




# linear sentences
sentence_1 = [Alice, loves, Bob]

# critical types but reducing with top of stack
sentence_2 = [Alice, whom, Chris, told, you, about, loves, Bob]

# one guarded critical fan
sentence_3 = [Alice, loves, Bob, who, Is, rich]

# two guarded critical fans (Italian language example)
sentence_4 = [Alice, cavalcava, cavallo, bianco, che, nitriva]
# 'Alice was riding a white horse who was neighing' (article omitted)

# example of grammatical sentence where a critical type is NOT guarded.
# This won't be parsed correctly by LinPP.
sentence_5 = [A, B, C, D, E]

# non sentence example
sentence_6= [loves, Bob, who]


positive_text = [sentence_1, sentence_2, sentence_3, sentence_4]


def test_reductions():
    red = parser(sentence_2)[0]
    result = {(1, 0): (0, 0), (3, 0): (2, 0), (3, 1): (1, 2), (4, 0): (3, 3),
              (4, 1): (3, 2), (5, 0): (4, 2)}

    assert (red == result), '(Alice loves Bob who is rich) not parsed with expected reductions'


def test_sentence_parsing():
    for sentence in positive_text:
        truth = is_sentence(sentence)
        assert (truth), 'sentence should have reduced to sentence type. False negative'




def test_nonsentence():
    value = is_sentence(sentence_6)
    assert (not value), 'nonsentence parsed to sentence type. False positive'




def test_counter_example():
    value = is_sentence(sentence_5)
    assert (not value)
