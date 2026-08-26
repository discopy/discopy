# TODO

Review round: cubic-dev-ai on #634, P2 on `.github/workflows/style-review.yml:4`.

> When a same-repository PR targeting a non-main branch opens ready with
> `TODO`, this run sets `wait=true`. Deleting `TODO` emits only
> `synchronize`; this workflow does not listen for it, and `no-todo-on-main`
> is restricted to `main`. The review remains skipped unless someone adds the
> manual label; handle `synchronize` so the waiting run can resume.

- [ ] review on the `synchronize` that deletes the `TODO` file, without
      reviewing every push and without firing twice on a main-based pull
      request, whose deleting push the guard turns into `ready_for_review`
- [ ] update the `CHANGELOG.md` entry
