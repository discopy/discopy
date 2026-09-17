# TODO

> so now the architecture is getting its own page? if so id rather have it
> as before as an html include/embedding

Keep the table embedded in the `discopy.abc` docstring, which needs the two
things that broke it fixed in place rather than sidestepped by a page:

- [ ] make the `:file:` of the `raw` directive resolve from a docstring
- [ ] make the table's links resolve from `_api/`, where the docstring renders
- [ ] drop `docs/architecture.rst` and its toctree entry
