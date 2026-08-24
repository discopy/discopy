"""P3: concrete categories satisfy their axioms, Matrix, Tensor, finset and python.
Miniature of the property over curated examples; red while its bullets (B1, B2, B3, B5, B6, B7) are live — issue #606.
"""
from discopy.matrix import Matrix
from discopy.tensor import Dim, Tensor
from discopy.python import additive, finset, multiplicative


def _additive_example():
    def inside(obj, tag=0):
        return (obj + 1, 2) if tag == 0 else (obj * 10, 0)
    # f : (int,) + (int,) -> (int, int) + (int,), tracing the last int
    return additive.Function(inside, dom=(int, int), cod=(int, int, int))


def test_p3():
    copy = Matrix.copy(2, 2)
    cases = [
        ("Matrix copy counit",
         lambda: copy >> Matrix.discard(2) @ Matrix.id(2) == Matrix.id(2)),
        ("Matrix copy cocommutative",
         lambda: copy >> Matrix.swap(2, 2) == copy),
        ("Matrix[bool] left trace of |0><0|",
         lambda: Matrix[bool]([1, 0, 0, 0], 2, 2).trace(1, left=True)
         == Matrix[bool]([0], 1, 1)),
        ("Tensor spider fusion 0-0 vs 0-1 >> 1-0 on Dim(2)",
         lambda: Tensor.spiders(0, 0, Dim(2))
         == Tensor.spiders(0, 1, Dim(2)) >> Tensor.spiders(1, 0, Dim(2))),
        ("finset swap(2, 1) block permutation",
         lambda: finset.Function.swap(2, 1) == finset.Function([2, 0, 1], 3, 3)),
        ("additive non-square trace value",
         lambda: _additive_example().trace(1)(5, 0) == 60),
        ("multiplicative trace of id (int, int)",
         lambda: multiplicative.Function.id((int, int)).trace()(5) == 5),
    ]
    failures = []
    for label, law in cases:
        try:
            if not law():
                failures.append(f"{label}: law violated")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    assert not failures, failures
