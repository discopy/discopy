# TODO

Human prompt, verbatim:

> ok now we received a lot of comments.
>
> first of all, we need to avoid using or introducing "supply", while parametrizing the e-hypergraphs like categories in @discopy/para.py   specializing methods specifically and adding a category named type parameter (first point of https://github.com/discopy/discopy/pull/730#issuecomment-6127590459)

Review round on [#730](https://github.com/discopy/discopy/pull/730):

> **@0x0f0f0f**, [things for this week](https://github.com/discopy/discopy/pull/730#issuecomment-6127590459):
> Using type parameters to distinguish between category instead of supply.

> **@toumix**, on `discopy/ehypergraph.py:301`: The terminology "supply" already
> exists with a different meaning (see Fong Spivak
> https://arxiv.org/abs/1908.02633) and it's used informally in the discopy docs
> already.
>
> As we discussed, let's avoid Boolean attributes and use a hierarchy of
> `EHypergraph` subclasses of `NamedGeneric["category"]` that reflect the
> hierarchy of structures instead, following the same template as the `para`
> module e.g. `ehypergraph.Symmetric` is the base class and
> `ehypergraph.Frobenius` is a subclass which assumes more structure of its
> underlying category.

> **@toumix**, on `discopy/ehypergraph.py:354`: we already have `discard` as
> method in other places let's avoid this for confusion

> **@toumix**, on `discopy/ehypergraph.py:356`: cartesian implies affine so this
> shouldn't do anything

> **@toumix**, on `discopy/ehypergraph.py:365`: I don't understand the `or not
> self.affine` part, from the docstring it seems like you mean
> `is_deterministic`?

> **@toumix**, on `discopy/abc.py:681`: We could also have `is_deterministic` for
> the copy-naturality alone, e.g. partial functions are deterministic but not
> necessarily affine. Then `is_cartesian` would become a property for the
> conjunction of affine and deterministic.

> **@toumix**, on `discopy/abc.py:689`: These shouldn't be a ClassVar, e.g. the
> identity and swaps are always affine and determinstic even though their class
> need not be.

> **@toumix**, on `discopy/markov.py:117`: We don't really want affine by
> default, we're using the terminology "Markov category" because the more
> general "copy-discard" or "garbage-sharing" would make ugly names for a
> module. We can add that comment in the docs somewhere.

> **@toumix**, on `discopy/utils.py:762`: This can move the abc module

> **@toumix**, on `discopy/ehypergraph.py:183`: Not sure what it means to share a
> box and an arity? Doesn't a box already have an arity? Seems like we store the
> arity as attributes, why don't we also store the box?

> **@toumix**, on `discopy/ehypergraph.py:192`: Let's add the condition this needs
> to respect in the docs explicitly, i.e. each element of the list should have
> length given by the sum of n_in and n_out.

> **@toumix**, on `discopy/ehypergraph.py:207`: It's not obvious why length should
> be stored as an attribute and not be just a property for the length of the
> columns, that design decision should be documented in the docstring.

> **@toumix**, on `discopy/ehypergraph.py:208`: Why do we have an empty row to
> begin with? I'm guessing it's because doubling from zero doesn't work but again
> it's not obvious let's document it.

> **@toumix**, on `discopy/ehypergraph.py:217`: I don't really like the idea of
> defining a category where the domain is always equal to the codomain, what
> we're dealing with is more faithfully described as a collection of monoids
> indexed by a pair of ints?

> **@toumix**, on `discopy/ehypergraph.py:243`: I don't think this deserves to be
> called `tensor`, it creates more confusion than anything: I would have
> expected tensor to act as addition on the arities but that's not the case.

> **@toumix**, on `discopy/ehypergraph.py:272`: I would rather avoid introducing a
> new set of abbreviations src and tgt which are synonymous to dom and cod. It
> doesn't make sense to call them `dom` and `cod` either because `cod == dom`
> doesn't make sense, but we could call them `doms` and `cods`?

- [x] Replace `Supply` by a hierarchy of e-hypergraphs parameterised by their
      `category`, as in `para`, each specialising the methods its structure
      calls for (`ehypergraph.py:301, 354, 356, 365`)
- [x] Drop `MarkovCategory.is_affine` and `is_cartesian`, which nothing reads
      once `Supply` is gone, so that `markov.Diagram` is no longer affine
      (`abc.py:681, 689`, `markov.py:117`)
- [ ] Move `Setoid` to `discopy.abc` (`utils.py:762`)
- [ ] Store the box of a `Shard` next to its arity (`ehypergraph.py:183`)
- [ ] Document that every row of a `Shard` has `n_in + n_out` wires
      (`ehypergraph.py:192`)
- [ ] Document why `Shard.length` is an attribute (`ehypergraph.py:207`)
- [ ] Document why the columns of a `Shard` start with one empty row
      (`ehypergraph.py:208`)
- [ ] Make `Shard` a monoid per arity rather than a category with
      `dom == cod` (`ehypergraph.py:217`)
- [ ] Rename `Shard.tensor` (`ehypergraph.py:243`)
- [ ] Rename `Shard.src` and `Shard.tgt` to `doms` and `cods`
      (`ehypergraph.py:272`)
- [ ] Update the changelog entry of #730
- [ ] `pflake8`, `pylint`, `pytest`
- [ ] Reply to and resolve the review threads of this round
