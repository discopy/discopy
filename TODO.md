# TODO

Human prompt (verbatim, user-confirmed 2026-08-27):
> 1. yes, i want this work to land in discopy/discopy as pull requests
> 2. yes, i am daydream6728 and you are nightmare6728, my agentic alter ego
> 3. no need, i just want the work to be available for review on the github interface, we'll refine the PRs there once they are open.

Context: this branch lands the quantum circuit carrier, stage of a 12-branch
split of daydream6728/discopy#2 ("The great purification: property matrix
over every carrier"), publishing pre-built, already-reviewed-and-tested work
from `daydream6728/discopy:split/4-quantum-circuit` onto `discopy/discopy` as
a ready-for-review pull request stacked on `split/4-tensor` (quantum.Circuit
subclasses tensor.Diagram).

- [x] Verify branch tip SHA against the fork before pushing
- [x] Push branch to discopy/discopy without rewriting any fetched commit
- [x] Open PR stacked on split/4-tensor using the repo's Why/What/How template
- [x] Link bugs fixed inline and file/link issues for bugs left open in BUGS.md
