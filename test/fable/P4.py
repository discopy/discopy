"""P4: gates are unitary and their gradients match finite differences.
Miniature of the property over curated examples; red while its bullets (B10, B11, B12) are live — issue #606.
"""
import numpy
import sympy

from discopy.quantum.gates import S, T, Controlled, CRz, Rz, Sqrt, U1


def _is_unitary(gate):
    array = numpy.asarray((gate >> gate.dagger()).eval().array, complex)
    n = int(round(array.size ** 0.5))
    return numpy.allclose(array.reshape(n, n), numpy.eye(n))


def _grad_matches_finite_difference(gate_factory, at=0.3, eps=1e-6):
    phi = sympy.Symbol('phi')
    grad = numpy.asarray(gate_factory(phi).grad(phi, mixed=False)
                         .lambdify(phi)(at).eval().array, complex)
    finite = (numpy.asarray(gate_factory(at + eps).eval().array, complex)
              - numpy.asarray(gate_factory(at - eps).eval().array, complex)
              ) / (2 * eps)
    return numpy.allclose(grad.flatten(), finite.flatten(), atol=1e-4)


def test_p4():
    unitaries = [("S", S), ("T", T),
                 ("Controlled(S)", Controlled(S)),
                 ("Controlled(T)", Controlled(T)),
                 ("CRz(0.3)", CRz(0.3)),
                 ("Controlled(Rz(0.3), distance=2)",
                  Controlled(Rz(0.3), distance=2))]
    cases = [(f"{label} unitary", lambda g=gate: _is_unitary(g))
             for label, gate in unitaries]
    cases += [
        ("Sqrt(1j) dagger is conjugate",
         lambda: numpy.allclose(complex(Sqrt(1j).dagger().eval().array),
                                numpy.conj(complex(Sqrt(1j).eval().array)))),
        ("U1 grad vs finite difference",
         lambda: _grad_matches_finite_difference(U1)),
        ("Rz grad vs finite difference",
         lambda: _grad_matches_finite_difference(Rz)),
    ]
    failures = []
    for label, law in cases:
        try:
            if not law():
                failures.append(f"{label}: law violated")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    assert not failures, failures
