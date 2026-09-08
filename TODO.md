# TODO

Review round on [#730](https://github.com/discopy/discopy/pull/730):

> **@daydream6728**, on `discopy/table.py:179`:
>
> probably fine to put it in `discopy.utils`? im sure we could make use of it
> in existing parts of the code: both cmap and hypergraph reimplement their own
> version for now.

- [x] Move `UnionFind` from `discopy/table.py` to `discopy/utils.py`
- [x] Use it in `cmap.CMap.from_glued`
- [x] Use it in `hypergraph.Hypergraph.is_boundary_connected`
- [x] Move its tests from `test/table.py` to `test/utils.py`
- [x] `pflake8`, `pylint`, `pytest`, coverage >= 98, sphinx
