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

Review feedback from @giodefelice on #505, quoted verbatim:

- [x] discopy/hopf.py:597 (`Double.star`) — "We need to find a better way to fix this.
      Definitely shouldn't be a method of Double. [...] This "star" should be obtainable from
      methods of pivotal, maybe also conjugate or dagger, and the definition of duals in Rep"
- [x] discopy/hopf.py:244 (`state` in `pivotal_element`) — "Let's avoid these function
      definitions inside methods. Can we make the calculations of gs so we don't need to
      reshape?"

Comment from @giodefelice on #484 (2026-07-30), quoted verbatim:

- [x] "1) merge main and get CI green."
- [x] "2) Make sure that the ribbon_element, pivotal_element and drinfeld_element are computed
      as single tensors (not tensor diagrams) as cached properties"
- [x] "3) Rewrite the doctest of Functor, draw the twist followed by the left trace of the
      braid, apply the functor get a tensor network with ribbon element and R matrix, check
      that the contraction is the identity."
