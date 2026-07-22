# TODO

Prompt ([#395](https://github.com/discopy/discopy/issues/395), verbatim):

> ## Problem
>
> When defining a custom symmetric category, it's natural to set a `swap_factory` to tell diagrams which `Swap` subclass to use (the README did exactly this). But **`swap_factory` is read nowhere in the codebase** — the swap machinery consults `braid_factory`:
>
> ```python
> # symmetric.py
> @classmethod
> def swap(cls, left, right):
>     return cls.braid(left, right)          # -> braided.Diagram.braid
>
> # braided.py
> @classmethod
> def braid(cls, left, right):
>     return hexagon(cls, cls.braid_factory)(left, right)   # reads braid_factory
> ```
>
> `grep -rn 'swap_factory' discopy/` returns nothing; `braid_factory` is the attribute set in every module (`symmetric.py:381` etc.).
>
> ## Why it bites
>
> Setting `swap_factory` assigns an attribute nobody reads, so `cls.swap(...)` silently falls back to the inherited base `Swap`. For a user-defined `@ar_factory` category, that base `Swap` is not an instance of the custom `Diagram` subclass, so downgrades that introduce swaps (e.g. `Hypergraph.to_diagram`, hence `Diagram.from_callable`) build a diagram whose swaps fail the category's `assert_isinstance` check:
>
> ```python
> from discopy.utils import ob_factory, ar_factory
> from discopy.symmetric import Ty, Box, Diagram, Swap
>
> @ob_factory
> class Ingredient(Ty): ...
>
> @ar_factory
> class Recipe(Diagram):
>     ob = Ingredient
>
> class CookingStep(Box, Recipe): ...
> class CookingSwap(Swap, CookingStep): ...
>
> Recipe.swap_factory = CookingSwap        # no-op
> egg, white, yolk = map(Ingredient, "xyz".replace("x","egg"))  # illustrative
> Recipe.swap(white, yolk)                 # -> symmetric.Swap, NOT a Recipe
> ```
>
> The failure surfaces far from the cause (a `TypeError: Expected Recipe, got symmetric.Diagram` deep inside `to_diagram`), with no hint that the `swap_factory` line did nothing. This was hit by the README's cooking example (fixed there by using `braid_factory`, see #386).
>
> ## Suggested fix
>
> Reduce the footgun, e.g. one of:
>
> - Add a `swap_factory` alias on `symmetric.Diagram` that `swap()`/`braid()` honour (or make `braid_factory` default from `swap_factory` when set), so the intuitive name works; and/or
> - Document on `symmetric.Diagram` that swaps come from `braid_factory`, and/or raise/warn when an unknown `*_factory` attribute is set on a `Diagram` subclass.
>
> Minimal footprint would be a `swap_factory` property that reads/writes `braid_factory`.

---

- [x] Add a `swap_factory` property on `symmetric.Diagram` that reads/writes `braid_factory`
- [x] Document on `symmetric.Diagram` that swaps come from `braid_factory`
- [ ] Test: a custom subclass setting `swap_factory` gets its own swaps through `Diagram.swap`, `Hypergraph.to_diagram` and `Diagram.from_callable` (README cooking example)
- [ ] Switch the README cooking example back to `swap_factory`
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`
