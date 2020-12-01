# -*- coding: utf-8 -*-

"""
Interface from spacy to discopy to create diagrams from a document.

Before using the interface, download a language model appropriate for
the task, see https://spacy.io/usage/models

>>> !python -m spacy download en_core_web_sm
>>> nlp = spacy.load("en_core_web_sm")
>>> doc = nlp("The quick brown fox jumped over the lazy dog.")

>>> !python -m spacy download fr_core_news_sm
>>> nlp = spacy.load("en_core_web_sm")
>>> doc = nlp("Le renard brun et agile a sauté par dessus le chien parasseux.")
"""

import functools

import spacy

from discopy import Ty, Word
from discopy.grammar import eager_parse


def assign_words(parsing, use_lemmas=False):
    """
    Given a parsing of some text (a list of spaCy tokens with punctuation removed), uses
    the part-of-speech tags and dependency structure to assign codomains to each word.
    
    The p-o-s tags constitute a set of basic types, and the codomains of the words are
    formed from tensor products of these basic types and their adjoints.
    
    In order to have a semantic output, a token with the 'ROOT' dependency tag is ensured
    to have a 'sentence' type in its codomain.
    
    Parameters
    ----------
    
    use_lemmas : bool
        use lemmas for names of the Word boxes e.g. 'loves' becomes Word('love', type)
        this is useful for simplifying a functor from sentence to circuit (we have less
        less words to specify the image of in the arrow mapping)
    """
    
    # a list of words and a set of basic types for this parsing
    words = []
    types = set()
    
    for token in parsing:
        
        # initial codomain
        ty = Ty('s') if token.dep_ == "ROOT" else Ty(token.pos_)
        # add to set of basic types (does nothing if type already in set)
        types.add(ty)

        # scan dependency tree and tensor codomain with appropriate left/right adjoints
        if token.n_lefts:
            lefts  = [Ty(token.pos_) for token in token.lefts if token in parsing]
            if len(lefts):
                ty = functools.reduce(lambda x,y: x @ y, lefts).r @ ty

        if token.n_rights:
            rights = [Ty(token.pos_) for token in token.rights if token in parsing]
            if len(rights): ty = ty @ functools.reduce(lambda x,y: x @ y, rights).l

        # add Word to list of Words in this parsing
        words.append(Word(token.lemma_ if use_lemmas else token.text, ty))
    
    return words, types


def sentence_to_diagram(parsing, **kwargs):
    """
    Takes a parsing of a sentence, assigns words to the tokens in the parsing, infers
    a set of basic types, and uses discopy's eager_parse to try and guess a diagram.
    """
    
    words, types = assign_words(parsing, **kwargs)
    
    try:
        diagram = eager_parse(*words)
    except:
        # this avoids an error if one of the sentences cannot be diagram-ed in
        # 'document_to_diagrams'
        print(f"Could not infer diagram from spaCy's tags: {parsing}")
        return None
    
    return diagram, types


def document_to_diagrams(doc, drop_stop=False, **kwargs):
    """
    Splits a document into sentences, and tries to create diagrams for each sentence,
    and a creates a set of all the basic types that were inferred from the document.

    Drops punctuation, and stop words if drop_stop=True.
    """
    
    # split sentences and parse
    sentences = [s for s in doc.sents]
    sentence_parsings = [[token for token in s if (token.pos != spacy.symbols.PUNCT) and not (drop_stop and token.is_stop)] for s in sentences]
    
    # get diagrams for each sentence and a set of basic types for the whole document
    diagrams, types = map(list,zip(*[sentence_to_diagram(parsing, **kwargs) for parsing in sentence_parsings]))
    types = functools.reduce(lambda x,y: x.union(y), types)
    
    return diagrams, types
