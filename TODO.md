# TODO

Cubic's correctness review of #608, quoted from its summary:

> - `.github/workflows/style-review.yml` executes PR-controlled `review.py` or `post.py` with `API_KEY` or `APP_TOKEN`, creating a direct secret-exfiltration and unauthorized-action risk [...]
> - `.github/style-review/review.py` silently drops changed files beyond the 400,000-character budget [...]
> - `.github/style-review/post.py` can mishandle malformed `findings.json`, while its GitHub API call has no timeout [...]
> - `.github/style-review/review.py` crashes on empty or non-JSON gateway responses and omits package `__init__.py` context [...]

plus a P3: `AGENTS.md`'s sign-off criterion still reads "every point of its
`TODO.md` is `[x]`" where the file is now deleted at that moment.

- [x] `imports` also
  resolves package `__init__.py` candidates
- [x] changed files
  dropped by the budget are named in the prompt like context files, not
  discarded silently
- [x] `post.py`'s
  GitHub calls get a bounded timeout
- [x] `post.py`
  rejects a `findings` payload that is not a list instead of reading it
  as clean
- [x] a gateway
  answer with no JSON raises a named error — declining the suggested
  fallback to clean, which reads a broken gateway as a clean diff
- [x] `AGENTS.md`
  sign-off criterion 1 reads the deletion: `TODO.md` is deleted, every
  point having been `[x]` or filed as an issue
- [x] decline the P0
  on the thread: same-repo authors already hold push access and the
  workflow file itself is branch-editable, forks are excluded — the trust
  boundary is who can push, not which ref the scripts load from
