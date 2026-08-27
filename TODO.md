# TODO

> “taken into account” and “open” are exclusive so remove “taken into account”
>
> make sure to tell the reviewer only to comment on the diff not on the whole
> file, its comments should be inline not a list of bullet points

- [x] The tally drops "taken into account": `N style remarks: X accepted /
      Y declined / Z still open`
- [x] `prompt.md` says to report only on a line the diff adds or changes,
      reading the whole file being for judging the change, not for finding
      things to say about the rest of it
- [x] `post.py` enforces it: a finding off the diff is dropped, not moved to
      the body, and the count of dropped ones is said
- [x] The review body carries no findings, every remark being an inline
      comment on its line
- [x] Closes discopy#673, which asked for exactly this ruling
