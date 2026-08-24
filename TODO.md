implement the to_compact proposal suggested by toumix on the PR thread. make a closed.Diagram.to_compact method implemented as a functor which turns a closed diagram with curry bubbles to a diagram with coeval morphisms.

- [x] Implement and verify `closed.Diagram.to_compact` as proposed.

implement to_compact on cmap as well and recover the curry diagram shapes in the doctest as they were on the main branch

- [x] Implement `CMap.to_compact` and restore the curry doctest shapes.

## Session: gh:discopy/discopy/pr/532

I reviewed your code and have the following comments. Please address them.

Reviewing pull request discopy/discopy#532: Align CMap on Hypergraph behavior
URL: https://github.com/discopy/discopy/pull/532
Head: 329c6b5d

## Local tuicr Comments

1. `discopy/biclosed.py:70` - this is calling CMap[biclosed.Diagram].to_compact which delegates to biclosed.Diagram.to_compact via roundtrip, but how can that work since only closed.Diagram.to_compact define it?
2. `discopy/closed.py:107-122` - didn't check to compare, but please ensure that this follows the same architecture as the DualRail functor.
3. `discopy/closed.py:220-228` - add a special case for isinstance(other, CMap)
4. `discopy/closed.py:234-241` - remove this function
5. `discopy/closed.py:244` - don't bind it like this, define to_compact in cmap.CMap and let inheritance do the work

## Existing GitHub Comments

### `discopy/hypergraph.py`

1. `discopy/hypergraph.py:1433` @toumix - I think we should only raise if both left and right are non-empty?
   <https://github.com/discopy/discopy/pull/532#discussion_r3814259954>
   - @daydream6728 - Agreed, `swap` now returns `id(left @ right)` when either side is empty, before asking for a symmetric category. Only the second call site can pass an empty side (`cod[j + 1:offset + i]` when `j == offset + i - 1`), and I could not build an input that reaches it: it needs the same wire twice in a box domain, which is non-monogamous and so already routed through `make_monogamous` above. So this is a guard rather than a bug fix, tell me if you would rather not carry the branch.

### `discopy/pivotal.py`

2. `discopy/pivotal.py:1` @toumix - No maps here?
   <https://github.com/discopy/discopy/pull/532#discussion_r3814268836>
   - @daydream6728 - Added `CMap = cmap.CMap[Diagram]`, with a `to_map` round-trip in `test_diagram_to_map_structure_and_errors`. Note `pivotal` has no `Hypergraph` alias either, and neither do `rigid` and `braided` — happy to add those too if you want them all uniform, I left them out of this PR.

### `docs/_static/cmap/biclosed-curry-left.dot`

3. `docs/_static/cmap/biclosed-curry-left.dot:1` @toumix - I can't check the `.dot` files from within github web GUI, trusting that they look good!
   <https://github.com/discopy/discopy/pull/532#discussion_r3814285281>
   - @daydream6728 - The two `.dot` baselines changed for one reason: the example now uses biclosed's own `Ty` and `Box` instead of importing `discopy.closed`, so the evaluation box draws as `>>` where `closed` drew `__call__`. Everything else in the files is identical.


   - @daydream6728 - > I can't check the .dot files from within github web GUI, trusting that they look good!

They don't:
<img width="168" height="301" alt="biclosed-curry-left" src="https://github.com/user-attachments/assets/62427786-38ba-48ed-b318-e2e2fa1a3727" />
on main:
<img width="220" height="521" alt="image" src="https://github.com/user-attachments/assets/7758fc09-e02c-4343-92a1-3148ac6893ee" />

In non-rigid categories, currying is represented as an explicit box instead of wiring. I will document how to interpret curry as wiring with eval and coeval.
   - @daydream6728 - I have another upcoming PR that draws bubbles in cmaps, but obviously this is changing very much from the way cmaps used to encode terms. I'm still not sure how to deal with it.
   - @daydream6728 - quickfix to offer a choice without touching much of the code: leave the generic implementation `cmap.CMap[closed.Diagram]` use `curry_factory`, but reintroduce the previous `class CMap(cmap.CMap[Diagram])` in closed.py that would override curry and uncurry to use wiring. 
   - @toumix - Maybe we can make that previous behaviour into a method `closed.Diagram.to_compact`? then we would get the bubbles with `closed.Diagram.to_map` and the wiring with `closed.Diagram.to_compact >> to_map`
   - @daydream6728 - yes i think that's a good way to do it, although it will likely break existing term-related PRs, implementing this as a functor then i think we're good to merge. 

- [WIP] @opencode-2026-08-24 16:39 Address the five local review comments and verify the existing threads remain satisfied.
