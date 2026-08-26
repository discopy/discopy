# TODO

> dis copy code review doesn't trigger when the PR is opened as ready, fix it

- [ ] make `style-review.yml` fire on a pull request opened already non-draft
- [ ] skip the review while a `TODO` file is still there, so the new trigger
      does not race `no-todo-on-main.yml`'s draft guard
- [ ] hand over to the correctness reviewer on the new trigger too
- [ ] add a `CHANGELOG.md` entry closing #615
