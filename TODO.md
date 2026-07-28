# TODO

> resurrect the branch, main got the rich display, so this PR should become just a small changeset to verify this works

Follow-up on #466, whose base #445 has landed on `main`. The branch is restarted
from `main` so that the diff is only the anywidget part.

- [x] restart the branch from `main` and drop everything already merged
- [x] add `discopy.drawing.widget.DiagramWidget`, an anywidget rendering an SVG
- [x] add `RichDisplay.to_widget` and include the widget view in the mimebundle
- [x] keep `_repr_mimebundle_` working when `anywidget` is not installed
- [x] add `anywidget` to the `docs` dependency group and to the api docs
- [x] add tests and run `pflake8 discopy` and `coverage run -m pytest`
