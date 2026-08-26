# TODO

> instead of full text then diff, it would make more sense to have the merge conflict format of git with `<<<<<<<HEAD` etc.?
>
> ha yes perfect let's go with unified diff style

toumix, on [#633](https://github.com/discopy/discopy/pull/633): drop the
separate `# Diff` section and the plain numbered-full-text "Changed"
section, replace both with one per-file listing that shows the whole new
file, unified-diff style — every line numbered by its new-file position,
a leading `+` for an added line, `-` (unnumbered, since it has no
new-file line) for a removed one.

- [ ] `review.py`: a function that gets git's own `-U100000` (full
      context) diff for one file and turns it into that numbered,
      inline-annotated listing — reuse git's diff algorithm rather than
      reimplementing it
- [ ] `changed_block`/`contents`/`assemble`: one block per changed file
      instead of two passes; drop `numbered()` (dead once `changed_block`
      stops needing it) and the `diff` parameter/`# Diff` section
- [ ] `style-review.yml`: pass `BASE_SHA` to the "Review the diff" step
      (already computed there for the `git diff` that builds `files.txt`)
- [ ] `prompt.md`: describe the new single-listing format
- [ ] `CHANGELOG.md` entry
- [ ] Smoke-test against a real modified file, a real added file, and the
      notebook (whose own cell fences are exactly the case this format
      needs to survive)
