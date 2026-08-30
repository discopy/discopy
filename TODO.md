Style review by `tencent/hy3` on pull request #688:

> Five levels of nested `for` loops (including the outer `for sine_sign`) violate DisCoPy's 'never nesting' guideline; consider extracting the B-check into a helper or using `itertools.product` to stay within three levels.

> The four `*_evaluate` helpers (here and at lines 514, 787, 1174) repeat the same one-liner `np.asarray(functor(diagram).array)`. Per 'DisCoPy never repeats itself', a small factory like `make_evaluate = lambda f: lambda d: np.asarray(f(d).array)` would avoid the duplication.

- [x] Extract or flatten the nested exact bialgebra check.
- [x] Deduplicate the notebook evaluator helpers.
- [x] Re-run notebook, symbolic, and lint checks.
