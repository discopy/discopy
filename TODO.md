# TODO

> **CodeRabbit review on 7db70f1 — actionable (discopy/monoidal.py:539)**
>
> Reject unsupported constraints instead of discarding them. `Testable.strategy`
> requires terminal overrides to declare only the constraints they implement.
> `Nat.strategy` and `Dim.strategy` accept `dom`/`cod` through `**_` but ignore
> them, then create values with transparent boundaries. `feedback.Ty.strategy`
> reaches `feedback.Wire.strategy` through inherited `monoidal.Ty.strategy`,
> which forwards each wire's `dom`/`cod`; `feedback.Wire.strategy` discards
> them, while `feedback.Wire.__init__` always creates transparent boundaries.
>
> **Nitpick (discopy/cat.py:329-335)** Expose the strategy phases as public
> classmethods: `extend`, `set_boundaries`, `from_dom`/`from_cod`, `traces`.
>
> **Nitpick (test/axioms.py:208-217)** Remove the duplicate assertions;
> `test_composable_shapes` already covers them.

Verified: asking `feedback.Ty.strategy` for a red boundary yields a wire with
`Colour('none')`, so the search can test a law on terms that do not meet the
constraint it was given. The finding is real and reachable.

- [WIP] @session_01SojhqwnPVHxjyJHhXs4hy1-2026-09-11 15:40 A terminal strategy
      refuses a constraint it cannot honour (`monoidal.Nat`, `monoidal.Dim`,
      `feedback.Wire`)
- [WIP] @session_01SojhqwnPVHxjyJHhXs4hy1-2026-09-11 15:40 Drop the duplicated
      assertions in `test/axioms.py`, and name `test_Small` after the
      `Subsingleton` it actually tests
- [ ] Decline the nested-phases nitpick on its thread, with the reason
- [ ] `pflake8` clean, both suites green, every change in the counts explained
