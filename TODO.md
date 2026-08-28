# Round of review feedback from USER on #676

> wait I thought we were keeping the off diff comments as an exception  that is
> allowed but discouraged (focus should be on the diff but there are cases where
> it makes sense to comment outside so it should go either inline when possible
> or in the comment body otherwise)
>
> another point: when the diff doesn't fit in context the style reviewer should
> mention it explicitly

- [x] a finding off the diff is posted, not dropped: inline wherever GitHub
      takes the comment — every line a hunk shows, added or context — and in the
      review body otherwise
- [x] `prompt.md` says the rule is the diff and that going outside it is allowed
      but discouraged, rather than saying such a finding is wasted
- [x] the review body says which changed files did not fit: reviewed from a
      diff only, or not reviewed at all
- [x] `CHANGELOG.md` and the #676 description say the amended rule, and #673
      hears about it
