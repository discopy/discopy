""" DisCoPy's property-testing module in action. """

from __future__ import annotations

from typing import Self

from hypothesis import find
from hypothesis.errors import NoSuchExample
from pytest import raises

from discopy import biclosed, cat, feedback, monoidal, traced
from discopy.axioms import (
    C0,
    C1,
    Atom,
    Axiom,
    AxiomFailure,
    Cells,
    Count,
    Equation,
    Goal,
    Level,
    Pair,
    Testable,
    Var,
    assert_axioms,
    axiom,
    connected,
    no_strategy,
)
from discopy.cat import Arrow, Box, Functor, Ob
from discopy.monoidal import Layer
from discopy.utils import AxiomError



def loops(equation: Equation) -> bool:
    """ The subspace of a law where every term is an endomorphism. """
    return all(term.dom == term.cod for term in equation.terms)


def test_axioms():
    assert_axioms(Arrow)

    class Classified(Arrow):
        """ A category with a broken law and an inapplicable one. """
        unitality = Arrow.unitality.failing("Never holds.")
        dagger_involution = Arrow.dagger_involution.inapplicable("No dagger.")

    assert_axioms(Classified)


def test_strategy():
    x, y = Ob('x'), Ob('y')
    find(Ob.strategy(), lambda ob: ob.name == "a")
    assert find(Arrow.strategy(dom=x, cod=x), lambda _: True) == Arrow.id(x)
    assert find(Arrow.strategy(dom=x, cod=y), lambda _: True).cod == y
    assert find(Arrow.strategy(dom=x), lambda _: True).dom == x
    assert find(Arrow.strategy(cod=y), lambda _: True).cod == y
    assert find(Box.strategy(dom=x), lambda _: True).dom == x


def test_annotated_sequents():
    """ A law draws its arguments from the sequents its annotations give. """
    @axiom
    def composing[A: C0, B: C0, C: C0](
            cls, f: C1[A, B], g: C1[B, C]) -> Equation:
        """ The domain of a composite. """
        return Equation(f.then(g).dom, f.dom)

    law = composing.bind(Arrow)
    f, g = find(
        law.pattern.strategy(Arrow), lambda args: all(a.inside for a in args))
    assert f.cod == g.dom and law(f, g)
    assert list(law.pattern.match((f, g))) == [
        {"A": f.dom, "B": f.cod, "C": g.cod}]
    assert find(law.strategy(), lambda equation: equation)
    x, y = Ob('x'), Ob('y')
    assert law(Box('f', x, y), Box('g', y, x))
    with raises(NoSuchExample):
        law.falsify()

    @axiom
    def atoms[X: Atom[C0], Y: Atom[C0], T: C0](
            cls, x: X, y: Y, t: T) -> Equation:
        """ Two atoms and a type. """
        return Equation(len(x @ y), 2)

    drawn = find(
        atoms.bind(monoidal.Diagram).pattern.strategy(monoidal.Diagram),
        lambda args: len(args[2]) > 1)
    assert [len(value) for value in drawn[:2]] == [1, 1]

    @axiom
    def shared[X: Atom[C0], A: C0, B: C0](
            cls, f: C1[X @ A, X @ B], x: X) -> Equation:
        """ A variable shared by an arrow and an object. """
        return Equation(f.dom[:1], x)

    law = shared.bind(monoidal.Diagram)
    f, x = find(law.pattern.strategy(law.category), lambda args: args[0].boxes)
    assert f.dom[:1] == x == f.cod[:1] and law(f, x)

    @axiom
    def delayed[A: C0, P: Pair[C0]](
            cls, f: C1[A @ P.d, A @ P], mem: P) -> Equation:
        """ The delay written as ``.d``. """
        return Equation(f.dom[-2:], mem.delay())

    law = delayed.bind(feedback.Diagram)
    f, mem = find(law.pattern.strategy(law.category), lambda args: args[0].boxes)
    assert len(mem) == 2 and law(f, mem)
    assert law.cells[0] is feedback.Ty and law.cells[1] is feedback.Diagram
    assert Cells.of(feedback.Diagram)[0] is feedback.Ty


def test_axiom_binding():
    assert repr(Axiom(lambda cls: NotImplemented)) == "Axiom(<lambda>)"
    assert eval(repr(Arrow.unitality)) == cat.Arrow.unitality
    assert hash(Arrow.unitality) == hash(eval(repr(Arrow.unitality)))
    assert Arrow.unitality != Functor.unitality
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented)()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).falsify()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).strategy()
    with raises(TypeError):
        Axiom(lambda cls: NotImplemented).cells
    assert axiom(lambda cls: NotImplemented).bind(Arrow)() is NotImplemented
    box = Box('f', Ob('x'), Ob('y'))
    assert Arrow.unitality(box)
    broken = Arrow.unitality.weaken(loops).failing("Never holds.")
    assert broken.subspace is loops
    loop = Box('g', Ob('x'), Ob('x'))
    with raises(AxiomFailure) as failure:
        broken(loop)
    assert failure.value.equation


def test_deferred_annotations():
    """ A law compiled without deferred annotations is refused. """
    namespace, source = {}, "def eager(cls, f: int): return NotImplemented"
    exec(compile(source, "<eager>", "exec", dont_inherit=True), namespace)
    with raises(TypeError, match="__future__"):
        axiom(namespace["eager"])


def test_inapplicable():
    law = Arrow.unitality.inapplicable("No identities to cancel.")
    assert law.name == "unitality"
    assert law.__doc__ == "No identities to cancel."
    assert law() is NotImplemented


def test_modulo():
    law = Arrow.unitality.modulo(lambda term: term.dom).bind(Arrow)
    assert law(Box('f', Ob('x'), Ob('y')))


def test_weaken():
    law = Arrow.unitality.weaken(loops).bind(Arrow)
    assert law.modulo(lambda term: term).subspace is loops
    equation = find(law.strategy(), lambda equation: equation.terms[1].inside)
    f = equation.terms[1]
    assert f.dom == f.cod and equation
    x, y = monoidal.Ty('x'), monoidal.Ty('y')
    box, scalar = monoidal.Box('f', x, y), monoidal.Box('s', x[:0], x[:0])
    assert connected(Equation(box, box))
    assert not connected(Equation(box, box @ scalar))
    with raises(NoSuchExample):
        find(monoidal.Diagram.bifunctoriality.strategy(), lambda eq: any(
            not term.is_boundary_connected for term in eq.terms))


def test_self_annotation():
    @axiom
    def absorbing(cls, f: Self) -> Equation:
        """ The identity on the domain absorbs into any arrow. """
        return Equation(cls.id(f.dom) >> f, f)

    law = absorbing.bind(Arrow)
    equation = find(law.strategy(), lambda _: True)
    assert isinstance(equation.terms[1], Arrow) and equation


def test_falsify():
    @axiom
    def trivial(cls, f: C1) -> Equation:
        """ Every arrow is an identity, which a box refutes. """
        return Equation(f, cls.id(f.dom))

    counterexample = trivial.bind(Arrow).falsify()
    assert isinstance(counterexample, Equation) and not counterexample
    assert counterexample.terms[0].inside
    assert not trivial.bind(Arrow).failing("Boxes exist.").falsify()
    with raises(NoSuchExample):
        Arrow.unitality.failing("Never holds.").falsify()
    with raises(NoSuchExample):
        Arrow.associativity.falsify()


def test_canonical():
    """ A law reads as a schema on the canonical arguments of its pattern. """
    A, B, C, D = map(Ob, "ABCD")
    f, g, h = Box('f', A, B), Box('g', B, C), Box('h', C, D)
    canonical = Arrow.associativity.canonical()
    assert canonical and canonical.terms[0] == f >> g >> h
    assert Arrow.associativity.pattern.canonical(Arrow) == (f, g, h)
    assert Arrow.unitality.canonical().terms[1] == Box('f', Ob('x'), Ob('y'))
    from discopy import symmetric
    swap = symmetric.Diagram.swap_inverse.canonical()
    assert swap and swap.terms[1] == symmetric.Id(
        symmetric.Ty('x') @ symmetric.Ty('y'))
    assert Arrow.unitality.failing("Never holds.").canonical()
    assert Arrow.unitality.inapplicable("No units.").canonical()\
        is NotImplemented
    with raises(TypeError):
        Arrow.unitality.inapplicable("No units.").draw()
    assert monoidal.Diagram.tensor_dom_typing.canonical()


def test_axioms_of_category():
    class Broken(Arrow):
        """ A category declaring an inherited law broken. """
        unitality = Arrow.unitality.failing("Never holds.")

    assert Broken.axioms["unitality"].broken
    assert not Arrow.axioms["unitality"].broken

    class Hidden(Arrow):
        """ Assigning a non-axiom over an inherited law drops it. """
        unitality = None

    assert "unitality" not in Hidden.axioms


def test_no_strategy():
    """ A class that does not generate its terms says so on `strategy`. """
    with raises(NotImplementedError) as err:
        Layer.strategy()
    assert "No search strategy implemented for Layer" in str(err.value)

    class Opted(Arrow):
        """ A class that would inherit a strategy for the wrong terms. """
        strategy = no_strategy

    with raises(NotImplementedError):
        Opted.strategy()
    assert Opted.axioms["unitality"] == Opted.unitality


def test_rules_are_inherited_and_bound():
    """ A category collects its rules through the MRO, like its axioms. """
    from discopy.axioms import Rule, search

    assert set(Arrow.rules) == {"id", "then", "box"}
    cut = Arrow.rules["then"]
    assert isinstance(cut, Rule) and cut.category is Arrow
    assert repr(cut) == "cat.Arrow.then" and cut.owner is not None
    assert str(cut.conclusion) == "A ⊢ C" and str(cut.pattern) == (
        "self: A ⊢ B, other: B ⊢ C")
    x, y = Ob('x'), Ob('y')
    arrow = find(
        search(Arrow, dom=x, cod=y, min_leaves=3, max_leaves=3,
               types=Ob.strategy()),
        lambda _: True)
    assert (arrow.dom, arrow.cod, len(arrow.inside)) == (x, y, 3)
    assert find(
        search(Arrow, dom=x, cod=x, max_leaves=0, types=Ob.strategy()),
        lambda _: True) == Arrow.id(x)


def test_generators():
    """ A class adjusts the generators its strategy invokes. """
    from discopy import rigid
    from discopy.axioms import Rule, invoked

    assert set(Arrow.generators) == {"box"}
    assert set(rigid.Diagram.generators) == {"box", "cups", "caps"}
    assert set(rigid.Diagram.rules) > set(rigid.Diagram.generators)
    x, y = Ob('x'), Ob('y')
    f, g = Box('f', x, y), Box('g', y, x)

    class Vocabulary(Arrow):
        """ Arrows over two boxes, no free one. """
        generators = {"f": Rule.constant(f), "g": Rule.constant(g)}

    assert [rule.name for rule in invoked(Vocabulary)] == [
        "id", "then", "f", "g"]
    assert Vocabulary.generators["f"].applies(Vocabulary, Goal.of(x, y, 1))
    assert not Vocabulary.generators["f"].applies(
        Vocabulary, Goal.of(y, x, 1))
    loop = find(
        Vocabulary.strategy(dom=x, cod=x, min_leaves=4),
        lambda arrow: len(arrow.inside) == 4)
    assert loop == f >> g >> f >> g
    bound = Vocabulary.generators["f"].bind(Vocabulary)
    assert find(bound.middles(x, x, Ob.strategy()), lambda _: True) == y
    with raises(NoSuchExample):
        find(Vocabulary.strategy(dom=x, cod=y), lambda arrow: any(
            box.name not in "fg" for box in arrow.inside))


def test_open_goals():
    """ A boundary left open is a hole the proof fills. """
    from discopy import frobenius, markov, rigid, symmetric
    from discopy.axioms import alignments

    x, y = symmetric.Ty('x'), symmetric.Ty('y')
    goal = Goal.of(x @ y, None, 1, symmetric.Ty())
    assert str(goal) == "x @ y ⊢ ?" and not goal.closed and goal.free(goal.cod)
    swapping = symmetric.Diagram.rules["permuting"]
    assert swapping.applies(symmetric.Diagram, goal)
    swapped = find(
        symmetric.Diagram.strategy(dom=x @ y, max_leaves=1),
        lambda diagram: any(
            isinstance(box, symmetric.Swap) for box in diagram.boxes))
    assert swapped.cod == y @ x
    cupping = rigid.Diagram.rules["cups"]
    x = rigid.Ty('x')
    aligned = alignments(
        cupping.conclusion, Goal.of(x @ x.r, None, 1, rigid.Ty()))
    assert len(aligned) == 1
    assert aligned[0][0]["X"].instantiate(aligned[0][1]) == x
    assert find(
        rigid.Diagram.strategy(dom=x @ x.r, max_leaves=1),
        lambda diagram: any(isinstance(box, rigid.Cup) for box in diagram.boxes)
    ) == rigid.Cup(x, x.r)
    x = markov.Ty('x')
    copied = find(
        markov.Diagram.strategy(dom=x, max_leaves=1),
        lambda diagram: any(
            isinstance(box, markov.Copy) for box in diagram.boxes)
        and diagram.cod == x @ x @ x)
    assert copied == markov.Copy(x, 3)
    x = frobenius.Ty('x')
    fused = find(
        frobenius.Diagram.strategy(dom=x @ x, max_leaves=1),
        lambda diagram: any(
            isinstance(box, frobenius.Spider) for box in diagram.boxes)
        and diagram.cod == x)
    assert fused == frobenius.Spider(2, 1, x)
    composed = find(
        Arrow.strategy(cod=Ob('y'), min_leaves=2, max_leaves=2),
        lambda arrow: True)
    assert composed.cod == Ob('y') and len(composed.inside) == 2


def test_leaf_applies_only_on_its_shape():
    """ A leaf is offered exactly when its pattern matches the sequent. """
    from discopy.axioms import generator, inapplicable, search

    class Toy(Arrow):
        """ Loops on every object, and no composition. """
        then = inapplicable("No composition.")(Arrow.then)

        @classmethod
        @generator
        def loop[A: C0](cls, dom: A) -> C1[A, A]:
            return Box('loop', dom, dom)

    assert set(Toy.rules) == {"id", "then", "box", "loop"}
    assert not Toy.rules["then"].applies(Toy, Goal.of(Ob('x'), Ob('x'), 2))
    x, y = Ob('x'), Ob('y')
    is_loop = lambda arrow: arrow.inside[0].name == 'loop'
    find(search(Toy, dom=x, cod=x, min_leaves=1, max_leaves=1,
                types=Ob.strategy()), is_loop)
    with raises(NoSuchExample):
        find(search(Toy, dom=x, cod=y, min_leaves=1, max_leaves=1,
                    types=Ob.strategy()), is_loop)


def test_rule_from_annotations():
    """ A rule reads its conclusion and premises off its annotations. """
    from discopy import traced
    from discopy.axioms import rule

    class Toy(traced.Diagram):
        """ Traced diagrams with a rule of their own. """

        @rule
        def looping[A: C0, B: C0, M: Atom[C0]](
                self: C1[M @ A, M @ B]) -> C1[A, B]:
            return self.trace(left=True)

    looping = Toy.rules["looping"]
    a, b = map(traced.Ty, "ab")
    unit = traced.Ty()
    assert looping.applies(Toy, Goal.of(a, b, 2, unit))
    assert not looping.applies(Toy, Goal.of(a, b, 1, unit))
    from discopy.axioms import search
    looped = find(
        search(Toy, dom=a, cod=b, min_leaves=2, max_leaves=2,
               types=traced.Ty.strategy(min_length=1)),
        lambda value: any(isinstance(box, traced.Trace) for box in value.boxes))
    assert (looped.dom, looped.cod) == (a, b)


def test_rules_from_patterns():
    """ A pattern rule matches, builds, and hints at the sequent it needs. """
    from discopy import braided, rigid, traced

    x, y, z = map(rigid.Ty, "xyz")
    cupping, unit = rigid.Diagram.rules["cups"], rigid.Ty()
    assert cupping.applies(rigid.Diagram, Goal.of(x @ x.r, unit, 1, unit))
    assert cupping.applies(rigid.Diagram, Goal.of(x.l @ x, unit, 1, unit))
    assert not cupping.applies(rigid.Diagram, Goal.of(x @ y.r, unit, 1, unit))
    assert not cupping.applies(rigid.Diagram, Goal.of(x @ x.r, unit, 2, unit))
    types = rigid.Ty.strategy()
    assert find(cupping.middles(x @ x.r, rigid.Ty(), types),
                lambda middle: not middle) == rigid.Ty()
    a, b, c = map(braided.Ty, "abc")
    braiding = braided.Diagram.rules["braid"]
    hinted = find(braiding.middles(a @ b @ c, c, types),
                  lambda middle: middle == b @ a @ c)
    assert hinted == b @ a @ c
    tracing, unit = traced.Diagram.rules["trace"], traced.Ty()
    assert tracing.applies(traced.Diagram, Goal.of(a, b, 2, unit))
    assert not tracing.applies(traced.Diagram, Goal.of(a, b, 1, unit))
    traced_arrow = find(
        traced.Diagram.strategy(dom=a, cod=b, min_leaves=2, max_leaves=2),
        lambda value: any(isinstance(box, traced.Trace) and box.left
                          for box in value.boxes))
    assert (traced_arrow.dom, traced_arrow.cod) == (a, b)


def test_pattern_matching():
    from discopy import rigid
    from discopy.axioms import Var, Word

    X, Y = Var.atom('X'), Var.atom('Y')
    A, B = Var.type('A'), Var.type('B')
    a, b, c = map(rigid.Ty, "abc")
    assert isinstance(X @ X.r, Word) and (X @ X.r).fixed
    assert str(X @ X.r) == "X @ X.r" and str(X @ X.r | Word()) == "X @ X.r | ()"
    assert str(X.l) == "X.l" and str(X.d) == "X.d" and str(X >> Y) == "(X >> Y)"
    assert list(X.match(a)) == [{"X": a}] and X.r.instantiate({"X": a}) == a.r
    assert not (A @ X).fixed
    assert list((X @ X.r).match(a @ a.r)) == [{"X": a}]
    assert list((X @ X.r).match(a @ b.r)) == []
    assert list((X @ X.l).match(a.r @ a)) == [{"X": a.r}]
    assert list((X.r @ X).match(a.r @ a)) == [{"X": a}]
    assert list((A @ X @ B).match(a @ b @ c)) == [
        {"A": rigid.Ty(), "X": a, "B": b @ c},
        {"A": a, "X": b, "B": c},
        {"A": a @ b, "X": c, "B": rigid.Ty()}]
    assert list((A @ A).match(a @ b @ a @ b)) == [{"A": a @ b}]
    assert list(Word().match(rigid.Ty())) == [{}]
    assert list((X @ Y).match(a)) == []
    x = cat.Ob('x')
    assert list(Word((A, )).match(x)) == [{"A": x}]


def test_pattern_exponentials_and_delays():
    from discopy import biclosed, feedback
    from discopy.axioms import Var

    B, E = Var.atom('B'), Var.atom('E')
    x, y = map(biclosed.Ty, "xy")
    assert list(((B << E) @ E).match((y << x) @ x)) == [{"B": y, "E": x}]
    assert list(((B << E) @ E).match((y << x) @ y)) == []
    assert list((E @ (E >> B)).match(x @ (x >> y))) == [{"E": x, "B": y}]
    M, A = Var.pair('M'), Var.type('A')
    u, v, w = map(feedback.Ty, "uvw")
    assert list((A @ M.delay()).match(w @ (u @ v).delay())) == [
        {"A": w, "M": u @ v}]
    assert list((A @ M).match(w @ u @ v)) == [{"A": w, "M": u @ v}]


def test_pattern_counts_and_choices():
    """ A count repeats an item and a boolean chooses a boundary. """
    from discopy import frobenius, rigid
    from discopy.axioms import Bool, Var, inapplicable

    X, Y = Var.atom('X'), Var.atom('Y')
    N, L = Var('N', Count()), Var('L', Bool())
    a, b = map(rigid.Ty, "ab")
    assert list((X ** N).match(a @ a)) == [{"N": 2, "X": a}]
    assert list((X ** N).match(a @ b)) == []
    assert list((X ** N).match(rigid.Ty())) == [{"N": 0}]
    assert (X ** N).instantiate({"X": a, "N": 3}) == a @ a @ a
    assert str(X ** N) == "X ** N"
    choice = L[X @ Y, Y @ X]
    assert str(choice) == "L[X @ Y, Y @ X]"
    assert list(choice.match(a @ b)) == [
        {"L": True, "X": a, "Y": b}, {"L": False, "Y": a, "X": b}]
    assert choice.instantiate({"L": False, "X": a, "Y": b}) == b @ a
    with raises(TypeError):
        X[X, Y]
    spiders = frobenius.Diagram.rules["spiders"]
    x, unit = frobenius.Ty('x'), frobenius.Ty()
    assert spiders.applies(frobenius.Diagram, Goal.of(x @ x, x, 1, unit))
    assert not spiders.applies(
        frobenius.Diagram, Goal.of(x @ x, x @ frobenius.Ty('y'), 1, unit))

    class Toy(Arrow):
        """ A category dropping the cut by an inapplicable override. """
        @inapplicable("Never composes.")
        def then(self, *others):
            return Arrow.then(self, *others)

    assert not Toy.rules["then"].applies(Toy, Goal.of(Ob('x'), Ob('x'), 2))
    assert Toy.rules["then"].__doc__ == "Never composes."


def test_two_categorical_patterns():
    """ A law of 2-categories names colours, types and diagrams by level. """
    from types import NoneType
    from discopy import abc, rigid
    from discopy.monoidal import Colour, Diagram, Ty, Wire

    law = Diagram.bifunctoriality
    assert law.owner is abc.TwoCategory
    assert law.cells.levels == (Colour, Ty, Diagram)
    assert Cells.of(rigid.Diagram, abc.Category).levels == (
        rigid.Ty, rigid.Diagram)
    assert Cells.of(rigid.Diagram, abc.TwoCategory).levels == (
        NoneType, rigid.Ty, rigid.Diagram)
    assert str(law.result) == "A @ C ⊢ U @ V"
    A = law.pattern.patterns["f"].dom.items[0]
    assert str(A.boundaries) == "X ⊢ Y" and A.level == 1 and A.kind == "type"
    assert [var.name for var in A.vars] == ["X", "Y", "A"]
    r, g = Colour("red"), Colour("green")
    x, y = Ty(Wire("x", r, g)), Ty(Wire("y", g, r))
    assert list(A.match(x)) == [{"X": r, "Y": g, "A": x}]
    tensor = Diagram.rules["tensor"]
    assert tensor.owner is abc.TwoCategory and tensor.boxes == 0
    matches = tensor.instances(Goal.of(x @ y, x @ y, 2, Ty()))
    assert sorted(len(env["A"]) for env in matches) == [0, 1, 2]
    assert all(env["A"].cod == env["B"].cod == env["C"].dom
               for env in matches)

    equation = find(
        Diagram.tensor_dom_typing.strategy(),
        lambda equation: equation.terms[0].dom != Ty().dom)
    assert equation and equation.terms[0].dom == equation.terms[1].dom
    assert Diagram.tensor_dom_typing.canonical()
    with raises(NoSuchExample):
        find(rigid.Diagram.tensor_dom_typing.strategy(),
             lambda equation: equation.terms[0].dom != rigid.Ty().dom)


def test_typechecking_on_call():
    """ A law checks its arguments and its equation against its patterns. """
    from discopy.monoidal import Box, Diagram, Ty

    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', y, x)
    assert Diagram.associativity(f, g, f)
    with raises(TypeError, match="expects"):
        Diagram.associativity(f, f, f)
    with raises(TypeError, match="expects"):
        Arrow.unitality(Ob('x'))

    @axiom
    def lying[A: C0, B: C0](cls, f: C1[A, B]) -> Equation[C1[A, A]]:
        """ States a sequent its terms do not have. """
        return Equation(f, f)

    with raises(TypeError, match="states"):
        lying.bind(Diagram)(f)
    assert lying.bind(Diagram)(Box('h', x, x))
    assert str(lying.pattern) == "f: A ⊢ B" and str(lying.result) == "A ⊢ A"
    assert lying.owner is None and lying.bind(Diagram).cells[0] is Ty


def test_pattern_instantiation():
    from discopy import rigid
    from discopy.axioms import Var, Word

    X, A = Var.atom('X'), Var.type('A')
    a, b = map(rigid.Ty, "ab")
    assert (A @ X @ X.r).instantiate({"A": a @ b, "X": a}) == a @ b @ a @ a.r
    assert Word().instantiate({}, unit=rigid.Ty()) == rigid.Ty()
    drawn = find(
        (X @ X.r).strategy(rigid.Diagram), lambda value: len(value) == 2)
    assert drawn[1:] == drawn[:1].r
    assert find((A @ X).strategy(rigid.Diagram, {"X": a}),
                lambda value: len(value) == 3)[2:] == a
