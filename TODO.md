# TODO

> on monday we merged a PR to implement adaptive rendering of SVG diagrams
> reacting to light/dark theme using media queries, so that we could have
> transparent backgrounds and keep drawings readable. However, in the marimo
> previews on the deployed documentation, the notebook theme does not react to
> the readthedocs theme switch. can you investigate and find a way to turn the
> notebook in dark theme when the user switches theme? because for now, the
> issue is that wires are turned white when the theme changes to dark, but the
> marimo background stays white so they become invisible.

- [x] Investigate why the marimo previews ignore the docs theme switch.
- [x] Make the exported notebooks follow the theme and turn dark with the docs.
- [WIP] @session_01PYmoyhxYhEXYSJSBMuPVrG-2026-09-09 08:55 Test the fix end-to-end in a browser, run the linter and the test suite.
- [x] Add a changelog entry.
