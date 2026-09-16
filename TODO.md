# TODO

> ideally there should be no other class defined in each syntax module besides Diagram, and the generators defined at this module immediately, instead of transitively.
> implement simplifications 1. and 2., and remove your generic hacky implementation of factories in utils, lets try with the manual method approach i described in my first prompt.
> you're right about the comment in symmetric, but still we shouldnt do Diagram.swap_factory = Swap, instead when cls = symmetric.Diagram the symmetric.Diagram.swap_factory should just return Swap directly instead of subclassing, and similar for all other generators

- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 `utils.generator`, a decorator declaring the factory of a generator as a method of the category introducing it, returning the class itself there and a subclass of the bases' factories on any other `@factory` class; `utils.Factory`, `utils.generators` and `utils.attributes` removed and `utils.factory` back to `cls.factory = cls`.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Every syntax module declares the generators it introduces or extends with `@generator()` in its `Diagram` body, no `Diagram.x_factory = X` assignment left; `compact.Diagram` loses its `trace_factory` line.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Roots initialise through `self.generator_factory.__init__`, the box of their own level: `feedback.Swap`, `Copy` and `Merge` keep only `delay`; `rigid.Box.z = 0` and the six `z = 0` of `ribbon` go; `ribbon.Functor` and `DualRail` recognise a `balanced.Braid`.
- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 Tests, README, CHANGELOG; `pflake8`, the full suite and the property tests green.
