import numpy as np

from discopy import ribbon, tensor
from discopy.tensor import Dim, Box
from discopy.hopf import HopfAlgebra, Representation, Intertwiner, Functor


# link diagrams, built inline from the ribbon generators where used
def _circle(x):
    return ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)


def _unlink(x):
    return _circle(x) @ _circle(x)


def _hopf_link(x):
    braid = ribbon.Braid(x, x)
    return (braid >> braid).trace(n=2)          # closure of sigma^2


# -- the Hopf algebra layer --------------------------------------------------

def test_group_algebra_is_valid():
    for n in [1, 2, 3, 5]:
        assert HopfAlgebra.cyclic(n).is_valid()
    # a non-cyclic group (Klein four) from its table
    table = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    assert HopfAlgebra.group_algebra(table).is_valid()


def test_double_is_quasitriangular_hopf_algebra():
    # the general double() applied to k[Z/n], not a hardcoded table
    for n in [2, 3]:
        D = HopfAlgebra.cyclic(n).double()
        assert D.dim == n * n
        assert D.is_valid()
        assert D.is_quasitriangular()


def test_commutativity_properties():
    # D(Z/2) is commutative and cocommutative; Sweedler's H4 is neither.
    D = HopfAlgebra.cyclic(2).double()
    assert D.is_commutative() and D.is_cocommutative()
    H4 = HopfAlgebra.sweedler()
    assert not H4.is_commutative() and not H4.is_cocommutative()


def test_double_of_sweedler():
    # Sweedler's H4 is neither commutative nor cocommutative and has S^2 != id,
    # so its double genuinely exercises the S^-1 in the double's multiplication
    # (a group algebra, being cocommutative with S^2 = id, would not).
    H4 = HopfAlgebra.sweedler()
    assert H4.is_valid() and H4.dim == 4
    Sarr = H4.antipode.eval(dtype=complex).array
    assert not np.allclose(Sarr @ Sarr, np.eye(4))     # S^2 != id
    D = H4.double()
    assert D.dim == 16
    assert D.is_valid() and D.is_quasitriangular()


def test_double_is_fine_grained():
    # the double's generators are composite diagrams, not materialised tensors:
    # every box acts on the small base object (size 2), never on D's size-4
    # space, so no N=4 -cubed structure tensor is ever formed.
    D = HopfAlgebra.cyclic(2).double()
    for gen in [D.mult, D.comult, D.unit, D.counit, D.antipode, D.R]:
        for box in gen.boxes:
            assert set(box.dom.inside) <= {2} and set(box.cod.inside) <= {2}


def test_no_r_matrix_paths():
    Z2 = HopfAlgebra.cyclic(2)
    noR = HopfAlgebra(
        Z2.unit, Z2.counit, Z2.mult, Z2.comult, Z2.antipode)
    assert noR.R is None
    assert noR.is_quasitriangular() is False
    assert 'quasitriangular' not in noR.validate()
    W = Representation.regular(noR)
    try:                             # no R-matrix: the braiding is undefined
        Intertwiner.braid(W, W)
        assert False
    except ValueError:
        pass


def test_representation_is_a_dim():
    # a Representation is a tensor.Dim carrying its (algebra, action)
    D, V = _double_and_module()
    assert isinstance(V, Dim) and V.dim == 2
    e, m = Representation.anyon(D, 0, -1), Representation.anyon(D, 1, 1)
    assert e != m                    # distinct 1-dim anyons differ by payload
    assert e != Dim(2)               # ... and from a bare Dim of another size
    assert len({e, m, V}) == 3       # hashable, all distinct
    payloadless = Representation(2)   # a product/adjoint: a Dim, no payload
    assert payloadless.action is None and payloadless.dim == 2
    from discopy import hopf, tensor  # noqa: F401  (used by eval)
    assert eval(repr(payloadless)) == payloadless


def test_repr_is_transparent():
    from discopy import hopf, tensor  # noqa: F401  (used by eval)
    H = HopfAlgebra.cyclic(2)
    assert eval(repr(H)) == H
    V = Representation.regular(H)
    assert eval(repr(V)) == V


# -- representations & structural morphisms ----------------------------------

def _double_and_module():
    D = HopfAlgebra.cyclic(2).double()
    e, m = Representation.anyon(D, 0, -1), Representation.anyon(D, 1, 1)
    return D, Representation.direct_sum([e, m])     # V = e (+) m


def test_representation_is_module():
    D, V = _double_and_module()
    assert V.is_module() and V.dim == 2
    assert Representation.regular(D).is_module()
    for anyon in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        assert Representation.anyon(D, *anyon).is_module()


def test_braiding_yang_baxter_and_inverse():
    _, V = _double_and_module()
    d = V.dim
    c = Intertwiner.braid(V, V).eval(dtype=complex).array
    c = c.reshape(d * d, d * d)   # input x output
    # braiding is invertible and not the swap
    assert not np.isclose(np.linalg.det(c), 0)
    swap = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            swap[a * d + b, b * d + a] = 1
    assert not np.allclose(c, swap)
    # the inverse braiding really inverts it
    ci = Intertwiner.braid(V, V, is_dagger=True)\
        .eval(dtype=complex).array
    ci = ci.reshape(d * d, d * d)
    assert np.allclose(c.T @ ci.T, np.eye(d * d))
    # Yang-Baxter on the braiding operator R = c^T (output x input)
    R, eye = c.T, np.eye(d)
    R12, R23 = np.kron(R, eye), np.kron(eye, R)
    assert np.allclose(R12 @ R23 @ R12, R23 @ R12 @ R23)


def test_quantum_dimension():
    _, V = _double_and_module()
    assert np.isclose(V.qdim(), 2)


def test_twist_from_braid():
    # the twist is the trace of the braid, defined without a ribbon element
    D = HopfAlgebra.cyclic(2).double()

    def theta(rep):
        return complex(
            Intertwiner.twist(rep).eval(dtype=complex).array)

    Vem = Representation.direct_sum(
        [Representation.anyon(D, 0, -1), Representation.anyon(D, 1, 1)])
    assert np.allclose(
        Intertwiner.twist(Vem).eval(dtype=complex).array, np.eye(2))
    assert np.isclose(theta(Representation.anyon(D, 1, -1)), -1)  # fermion
    assert np.isclose(theta(Representation.anyon(D, 0, 1)), 1)    # vacuum


def test_is_module_rejects_non_module():
    Z2 = HopfAlgebra.cyclic(2)
    ty = Z2.ty
    zero = Box('bad', ty @ Dim(2), Dim(2), np.zeros((2, 2, 2)))
    assert not Representation(
        algebra=Z2, action=zero).is_module()          # rho(1) != id


def test_snake_equations():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    identity = F(ribbon.Id(x)).eval(dtype=complex).array
    assert np.allclose(F(left).eval(dtype=complex).array, identity)
    assert np.allclose(F(right).eval(dtype=complex).array, identity)


# -- the functor and the topological invariant -------------------------------

def test_functor_returns_a_tensor_network():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    network = Functor(ob_map={x: V}, ar_map={})(_hopf_link(x))
    assert isinstance(network, tensor.Diagram)      # not a contracted Tensor
    # the user contracts it themselves with .eval
    assert np.isclose(complex(network.eval(dtype=complex)), 0)


def test_reidemeister_moves():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    X = ribbon.Ty('x')
    r2 = ribbon.Braid(X, X) >> ribbon.Braid(X, X).dagger()
    identity = F(ribbon.Id(X @ X)).eval(dtype=complex).array
    assert np.allclose(F(r2).eval(dtype=complex).array, identity)
    lhs = ribbon.Braid(X, X) @ X >> X @ ribbon.Braid(X, X) \
        >> ribbon.Braid(X, X) @ X
    rhs = X @ ribbon.Braid(X, X) >> ribbon.Braid(X, X) @ X \
        >> X @ ribbon.Braid(X, X)
    assert np.allclose(F(lhs).eval(dtype=complex).array,
                       F(rhs).eval(dtype=complex).array)


def test_nontrivial_link_invariant():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    circle = complex(F(_circle(x)).eval(dtype=complex))
    unlink = complex(F(_unlink(x)).eval(dtype=complex))
    hopf = complex(F(_hopf_link(x)).eval(dtype=complex))
    assert np.isclose(circle, 2)       # unknot -> qdim
    assert np.isclose(unlink, 4)       # 2 unknots
    assert np.isclose(hopf, 0)         # Hopf link
    assert not np.isclose(hopf, unlink)


def test_crossing_number_distinguishes_closures():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    X = ribbon.Ty('x')
    closures = [
        ribbon.Id(X @ X).trace(n=2),                             # unlink
        ribbon.Braid(X, X).trace(n=2),                           # unknot
        (ribbon.Braid(X, X) >> ribbon.Braid(X, X)).trace(n=2),   # Hopf link
    ]
    values = [complex(F(c).eval(dtype=complex)) for c in closures]
    assert np.allclose(values, [4, 2, 0])


def test_two_colour_mutual_braiding():
    D = HopfAlgebra.cyclic(2).double()
    e = Representation.anyon(D, 0, -1)
    m = Representation.anyon(D, 1, 1)
    xe, xm = ribbon.Ty('e'), ribbon.Ty('m')
    F = Functor(ob_map={xe: e, xm: m}, ar_map={})

    def value(a, b):
        link = (ribbon.Braid(a, b) >> ribbon.Braid(b, a)).trace(n=2)
        return complex(F(link).eval(dtype=complex))

    assert np.isclose(value(xe, xm), -1)   # mutual statistics -1
    assert np.isclose(value(xe, xe), 1)
    assert np.isclose(value(xm, xm), 1)


def test_functor_on_generic_box():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    box = ribbon.Box('f', x, x)
    F = Functor(ob_map={x: V}, ar_map={box: np.array([[1, 2], [3, 4]])})
    assert np.allclose(
        F(box).eval(dtype=complex).array, [[1, 2], [3, 4]])


def test_functor_on_type():
    _, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    assert F(x) == Dim(2)
    assert F(x @ x) == Dim(2, 2)


def test_twist_in_diagram():
    # a group algebra: trivial twist, but exercises the Twist functor path
    Z2 = HopfAlgebra.cyclic(2)
    V = Representation.regular(Z2)
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={})
    twist = ribbon.Twist(x)
    assert np.allclose(
        F(twist).eval(dtype=complex).array, np.eye(2))
    assert np.allclose(
        F(twist.dagger()).eval(dtype=complex).array, np.eye(2))
