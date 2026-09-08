# TODO

Review feedback from @daydream6728 on #744:

> `tree_keys` isn't a very good name because this value controls all serialisation methods and not just json. what about `serialised_attrs`?

> remove these functions, they are useless. if we want to serialize a collection, just do a for loop.

> just use super() here, if super() doesn't work out of the box it might hide deeper issues.

- [x] rename `tree_keys` to `serialised_attrs` everywhere, `CHANGELOG.md` included
- [x] delete `utils.encode` and `utils.decode`, inlining a for loop in `to_tree` and `from_tree`
- [x] `cat.Box.__repr__` calls plain `super()`: `Arrow.__repr__` learns that a box is its own generator, closing the cycle
- [x] `pflake8` and `pytest` green, threads replied and resolved
