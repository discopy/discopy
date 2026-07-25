get the PR about rich display rendering and make another PR fixing alexis comment and conflict

Alexis on #445: "should be rebased on the svg PR which should be rebased on
main which already has the script for generating docs in CI". The SVG PR #463
and the determinism fixes #457 / #469 are now on `main`, so this branch starts
from `main` and replays the rich-display feature on top of it, which also
resolves the conflict in `discopy/drawing/backend.py` and drops the stale
`docs/_static` churn.

Mathematical design: rich display is a functor from diagrams to images that
factors through `Drawing`, i.e. anything with a `to_drawing` method renders to
the SVG and PNG mimetypes. Rendering must be deterministic: equal diagrams give
byte-for-byte equal images, so the figure metadata and the SVG hash salt are
fixed rather than taken from the environment.

- [ ] Replay the `RichDisplay` mixin on top of `main`
- [ ] Let `savefig` take an explicit `format` so drawings render to a buffer
- [ ] Add the rich-display tests and check `pflake8` and the test suite
