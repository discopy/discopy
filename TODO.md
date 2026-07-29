# TODO

Review feedback from @giodefelice on #484, quoted verbatim (one point per comment, per AGENTS.md):

- [x] discopy/hopf.py:203 (`pivotal_element`) — "I'm pretty sure
      this method can be made shorter and more efficient. Is there a shortcut to solving the
      system and then searching grouplike elements?"
- [x] discopy/hopf.py:257 (`is_ribbon` closure in
      `pivotal_element`) — "This looks like duplicate code with ribbon_element. We can add a
      method is_ribbon to Algebra and use it in both."
- [x] discopy/hopf.py:516 (`Algebra.taft` docstring) — "What is g
      when these are functions of x??"
- [x] discopy/hopf.py:592 (`Double.__init__`, the `star` lambda)
      — "This looks strange very strange. this be done with transposition?"
- [x] discopy/hopf.py:1034 (`Functor.__call__`, the `ribbon.Cap`
      branch) — "Isn't this already handled in the standard ribbon.Functor?"
