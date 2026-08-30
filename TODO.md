# PR #688 review round

- [x] Address Cubic's exact-checker finding: "The exact checker does not
  establish that the displayed B witness is actually nonzero at an admissible
  EU root. Assert an exact root interval together with the `s > 0` branch and
  prove `expected > 0` there, or evaluate an isolated root before declaring
  the countermodel verified."
- [x] Address Cubic's rendering finding: "The inline math formula in the
  bialgebra countermodel section is missing a backslash: it reads
  `X_{10,10,0}=\\frac14,qquad` instead of `\\qquad`."
- [x] Run the exact checker, notebook export check, lint, and diff checks.
- [ ] Reply to and resolve both review threads after pushing the fixes.
