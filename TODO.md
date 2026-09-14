Let's get this abstract categorial grammar framework properly implemented, make sure to read the paper first https://github.com/discopy/discopy/pull/400

(The paper: Philippe de Groote, *Towards Abstract Categorial Grammars*, ACL 2001.)

Review round of 2026-09-14 on the pull request, toumix:

> why do we need this one? (`discopy/grammar/abstract.py:105`, `Curry.ob = Ty`)

> same here and below, we shouldn't need them right? (`discopy/grammar/abstract.py:126`, `Trace.ob = Ty`, `Sum.ob = Ty`)

> That's not the definition of ACG, Lexicon should be a functor from abstract to abstract grammar.
>
> Then the catch is that from any categorial grammar diagram we get an abstract diagram together with a string-valued Lexicon, i.e. with a single generating type X and constants X to X for each token so that string concatenation is composition of constants.

> This test doesn't make much sense let's replace it with the one inspired by the higher-order discocat notebook, i.e. we generate random predicates over a finite-size universe and then check that the natural language sentence compiles to the same value as the Python one `all(not Woman[x] or any(Man[y] and Married[x, y] for y in U) for x in U)` vs `any(Song[x] and all(not Child[y] or Learnt[x, y] for y in U) for x in U)`

wangdaphne:

> The `eval()` also needs to be checked here. Actually both `lexicon(every_woman_married_a_man).eval()` and `lexicon(every_child_learnt_a_song).eval()` would fail since the definition of the quantifier "A" is not linear (requires to duplicate the variable $y$; so the diagram cannot be evaluated).

> Found a bug there: diagram.is_linear evaluates to True here (even if the variable v is copied). Apparently that comes from the class Box. In closed.py:215: `is_linear = True` automatically sets is_linear=True without looking into the box); but diagram.arg.is_linear evaluates to False.

- [x] Merge `main` into the branch (`CHANGELOG.md` conflict).
- [ ] Drop the `ob = Ty` on `Curry`, `Trace` and `Sum` if the factory closure holds without them, answer both threads.
- [ ] `Lexicon` is a functor between abstract grammars (de Groote 2001, §2.2): atomic types to types, constants to terms; the string vocabulary of §4 (one atomic type, words as constants of type `* >> *`, concatenation as composition) and the string lexicon a categorial grammar comes with.
- [ ] Replace `test_python_Functor_on_terms` with the finite-universe test: random predicates, the two sentences compile to the same truth value as the Python expressions.
- [ ] Reproduce and fix `eval()` on the semantic lexicon's non-linear images, the new test evaluates them.
- [ ] Reproduce and fix `is_linear` on `Curry`, `Trace`, `Sum` and on terms.
- [ ] `CHANGELOG.md`, docstrings, `pflake8` and the full suite green, then delete `TODO.md`.
