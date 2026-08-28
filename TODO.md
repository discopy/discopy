# TODO

> i opened a parallel PR #645 on refactoring the workflows, please merge it into
> this one
>
> there's also 617 and 620 that are related to style review, let's make just two
> PRs: 645 with the workflow refactor (easy to merge) and one big with this one
> and the other two, start from a fresh PR so we can see the style reviewer in
> action on something sizeable

- [x] Branch afresh off `main` and merge #645, so this sits on the refactored
      workflows and can use their test harness
- [x] Merge #672 (the reviewer's memory), #617 (degrade past the budget) and
      #620 (the discussion in the prompt)
- [x] Reconcile #620's `thread.py` with #672's `history.py`: one module reads
      the three listings, one renders the transcript, neither fetches twice
- [x] Reconcile #617's degrade with #672's ordering: the revision under review
      is budgeted first and rendered last
- [x] Move the style reviewer's own tests into `.github/tests`, where #645 put
      the harness, and give `history.py` and `thread.py` their own
- [ ] Close #617, #620 and #672 as superseded, leaving #645 to merge on its own
