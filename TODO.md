# TODO

> dis copy code review doesn't trigger when the PR is opened as ready, fix it

- [x] make `style-review.yml` fire on a pull request opened already non-draft
- [x] skip the review while a `TODO` file is still there, so the new trigger
      does not race `no-todo-on-main.yml`'s draft guard
- [x] hand over to the correctness reviewer on the new trigger too
- [x] add a `CHANGELOG.md` entry closing #615
