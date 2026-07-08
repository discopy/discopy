<!--
Thank you for contributing to DisCoPy! Please keep this description short.
See CONTRIBUTING.md for our philosophy, code style guide and LLM guidelines.
-->

## What

<!-- The behaviour that is preserved, and why the previous abstraction
level was wrong. Per CONTRIBUTING.md this part must be written by a
human or quote a human's prompt verbatim. -->

## Changes

<!-- What moved, merged or got simplified. This part may be LLM-generated. -->

-

## Checklist

- [ ] This change is behaviour-preserving: no test assertions were changed
- [ ] Any duplicated logic now lives once, at the right level of abstraction
- [ ] Nesting stays within three levels
- [ ] `uv run pflake8 discopy` and `uv run coverage run -m pytest` then `uv run coverage report -m` all pass
- [ ] I have respected the [code style guide](../../CONTRIBUTING.md#code-style-guide)
- [ ] Any contribution from a large language model is explicitly indicated as such
