"""Examples from de Groote (2001), checked by interpreting string functions."""

from discopy.grammar.abstract import Functor, Lexicon, Position, String, Ty
from discopy.python import Function


def tokens(term):
    python = Functor(
        ob_map={Position: list},
        ar_map=lambda word: lambda: lambda xs: xs + [word.name],
        cod=Function)
    return python(term)()([])


def context_free_grammar():
    S = Ty("S")
    A, B = S("A"), (S >> S)("B")
    lexicon = Lexicon(ob_map={S: String}, ar_map={
        A: Position(lambda x: x),
        B: String(lambda x: String("a").then(x, String("b")))})
    return A, B, lexicon


def test_context_free_language():
    """Section 4.1: S -> epsilon | a S b generates a^n b^n."""
    term, B, lexicon = context_free_grammar()
    for n in range(5):
        image = lexicon(term)
        assert term.is_linear and image.is_linear
        assert image.cod == String and not image.freevars
        expected = ["a"] * n + ["b"] * n
        assert tokens(image) == tokens(image.normal_form()) == expected
        term = B(term)


def test_tree_adjoining_language():
    """Section 4.3: adjunction generates a^n b^n c^n d^n."""
    S, Sp, Spp = map(Ty, ("S", "S'", "S''"))
    auxiliary = Spp >> Sp
    A, B, C = ((auxiliary >> S)("A"),
               (Spp >> (auxiliary >> Sp))("B"), auxiliary("C"))
    a, b, c, d = map(String, "abcd")
    lexicon = Lexicon(
        ob_map={S: String, Sp: String, Spp: String},
        ar_map={
            A: (String >> String)(lambda f: f(Position(lambda x: x))),
            B: String(lambda x: (String >> String)(
                lambda g: a.then(g(b.then(x, c)), d))),
            C: String(lambda x: x)})
    adjunct = C
    for n in range(5):
        term = A(adjunct)
        image = lexicon(term)
        assert term.cod == S and not term.freevars and term.is_linear
        assert image.cod == String and not image.freevars and image.is_linear
        expected = [letter for letter in "abcd" for _ in range(n)]
        assert tokens(image) == tokens(image.normal_form()) == expected
        previous = adjunct
        adjunct = Spp(lambda x: B(x)(previous))


def test_lexicon_composition_on_terms():
    """Section 6: composition substitutes a second lexicon into the first."""
    A, B, first = context_free_grammar()
    second = Lexicon(ob_map={Position: Position}, ar_map={
        String("a"): String("A") >> String("A"),
        String("b"): String("B")})
    term = B(B(A))
    sequential, composed = second(first(term)), (first >> second)(term)
    assert sequential.normal_form().alpha_equivalent(composed.normal_form())
    assert tokens(sequential) == tokens(composed) == list("AAAABB")


def test_shared_abstract_language_preserves_ambiguity():
    """Shared surface form, distinct meanings for two abstract readings."""
    n, s, e, t = map(Ty, ("n", "s", "e", "t"))
    river, finance = n("river-bank"), n("financial-bank")
    visit = (n >> s)("visit")
    syntax = Lexicon(ob_map={n: String, s: String}, ar_map={
        river: String("bank"), finance: String("bank"),
        visit: String(lambda x: String("visit") >> x)})
    RIVER, FINANCE, VISIT = e("RIVER"), e("FINANCE"), (e >> t)("VISIT")
    semantics = Lexicon(ob_map={n: e, s: t}, ar_map={
        river: RIVER, finance: FINANCE, visit: VISIT})
    readings = visit(river), visit(finance)
    forms = [syntax(reading).normal_form() for reading in readings]
    meanings = [semantics(reading) for reading in readings]
    assert readings[0] != readings[1]
    assert forms[0].alpha_equivalent(forms[1])
    assert tokens(forms[0]) == tokens(forms[1]) == ["visit", "bank"]
    assert meanings == [VISIT(RIVER), VISIT(FINANCE)]
    python = Functor(ob_map={e: str, t: str}, ar_map={
        RIVER: lambda: "river bank", FINANCE: lambda: "financial bank",
        VISIT: lambda: lambda place: f"visit({place})"}, cod=Function)
    assert [python(meaning)() for meaning in meanings]\
        == ["visit(river bank)", "visit(financial bank)"]
