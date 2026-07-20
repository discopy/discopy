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


def test_double_antipode_inverse():
    # the double's S^-1 = u^-1 S(x) u comes from the Drinfeld element,
    # a composite of the double's own generators — check it inverts S,
    # also when S_D^2 != id (the double of sweedler)
    for base in [HopfAlgebra.cyclic(2), HopfAlgebra.sweedler()]:
        D, n = base.double(), base.dim ** 2
        # contract the pieces separately to stay within einsum's symbol limit
        S = D.antipode.eval(dtype=complex).array.reshape(n, n)
        mult = D.mult.eval(dtype=complex).array.reshape(n, n, n)
        swap = tensor.Diagram.swap(D.ty, D.ty)
        u = (D.R >> tensor.Diagram.id(D.ty) @ D.antipode >> swap
             >> D.mult).eval(dtype=complex).array.reshape(n)
        u_inv = (D.R >> (D.antipode >> D.antipode)
                 @ tensor.Diagram.id(D.ty) >> swap
                 >> D.mult).eval(dtype=complex).array.reshape(n)
        Si = np.einsum('xy,i,iyk,j,kjl->xl', S, u_inv, mult, u, mult)
        assert np.allclose(Si @ S, np.eye(n))
        assert np.allclose(S @ Si, np.eye(n))


def test_double_is_fine_grained():
    # the double's generators are composite diagrams, not materialised tensors:
    # every box acts on the small base object (size 2), never on D's size-4
    # space, so no N=4 -cubed structure tensor is ever formed.
    D = HopfAlgebra.cyclic(2).double()
    for gen in [D.mult, D.comult, D.unit, D.counit, D.antipode, D.R,
                D.antipode_inv]:
        for box in gen.boxes:
            assert set(box.dom.inside) <= {2} and set(box.cod.inside) <= {2}


def test_no_r_matrix_paths():
    Z2 = HopfAlgebra.cyclic(2)
    noR = HopfAlgebra(
        Z2.unit, Z2.counit, Z2.mult, Z2.comult, Z2.antipode)
    assert noR.R is None
    assert noR.is_quasitriangular() is False
    assert noR.is_valid()
    W = Representation[noR].regular()
    try:                             # no R-matrix: the braiding is undefined
        Intertwiner[noR].braid(W, W)
        assert False
    except ValueError:
        pass
    try:                             # no algebra at all: same error
        Intertwiner.braid(W, W)
        assert False
    except ValueError:
        pass
    assert W.l.is_module()           # the inverse antipode is computed


def test_non_invertible_antipode_is_flagged():
    Z2 = HopfAlgebra.cyclic(2)
    ty = Z2.ty
    zero = Box('S', ty, ty, np.zeros((2, 2)))
    try:
        HopfAlgebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, zero)
        assert False
    except ValueError:
        pass


def test_class_generic_over_the_algebra():
    # Representation[H] and Intertwiner[H] carry the algebra on the class,
    # accessible from instances, and parametrisation is cached by equality
    D = HopfAlgebra.cyclic(2).double()
    assert Representation[D] is Representation[HopfAlgebra.cyclic(2).double()]
    assert Representation[D].algebra == D
    V = Representation[D].regular()
    assert V.algebra == D
    braid = Intertwiner[D].braid(V, V)
    assert Intertwiner[D].algebra == D and braid.algebra == D


def test_representation_is_a_dim():
    # a Representation is a tensor.Dim carrying its action
    D, V = _double_and_module()
    assert isinstance(V, Dim) and V == Dim(2)
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert e.action != m.action     # anyons are distinguished by their action
    assert e != Dim(2)              # equality is that of the underlying Dim
    payloadless = Representation(2)  # no algebra: a Dim without an action
    assert payloadless.action is None and payloadless == Dim(2)
    assert hash(payloadless) == hash(Dim(2))    # consistent with equality
    assert Representation(2, 3).r == Representation(2, 3).l == Dim(3, 2)
    assert Representation() @ Representation() == Dim(1)   # the unit
    from discopy import hopf, tensor  # noqa: F401  (used by eval)
    assert eval(repr(payloadless)) == payloadless


def test_repr_is_transparent():
    from discopy import hopf, tensor  # noqa: F401  (used by eval)
    H = HopfAlgebra.cyclic(2)
    assert eval(repr(H)) == H
    V = Representation[H].regular()
    assert eval(repr(V)) == V and eval(repr(V)).action == V.action


# -- representations & structural morphisms ----------------------------------

def _double_and_module():
    D = HopfAlgebra.cyclic(2).double()
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    return D, Representation[D].direct_sum([e, m])     # V = e (+) m


def test_representation_is_module():
    D, V = _double_and_module()
    assert V.is_module() and V == Dim(2)
    assert Representation[D].regular().is_module()
    for anyon in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        assert Representation[D].anyon(*anyon).is_module()


def test_tensor_of_representations():
    # the product of modules acts through the comultiplication
    D, V = _double_and_module()
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert (e @ m).is_module() and (e @ m).action is not None
    VV = V @ V
    assert VV == Dim(2, 2) and VV.is_module()
    assert (V @ e @ m).is_module()       # n-ary, coassociative


def test_braiding_yang_baxter_and_inverse():
    D, V = _double_and_module()
    d = 2
    c = Intertwiner[D].braid(V, V).eval(dtype=complex).array
    c = c.reshape(d * d, d * d)   # input x output
    # braiding is invertible and not the swap
    assert not np.isclose(np.linalg.det(c), 0)
    swap = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            swap[a * d + b, b * d + a] = 1
    assert not np.allclose(c, swap)
    # the inverse braiding really inverts it
    ci = Intertwiner[D].braid(V, V, is_dagger=True)\
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


def test_dual_representation():
    # the right dual carries the antipode-twisted action rho(S h)^T,
    # also when S^2 != id (sweedler) and for structured types (regular)
    D, V = _double_and_module()
    assert V.r.is_module()
    assert Representation[HopfAlgebra.sweedler()].regular().r.is_module()
    n = D.dim
    rho = V.action.eval(dtype=complex).array.reshape(n, 2, 2)
    S = D.antipode.eval(dtype=complex).array.reshape(n, n)
    dual = V.r.action.eval(dtype=complex).array.reshape(n, 2, 2)
    assert np.allclose(dual, np.einsum('hk,kio->hoi', S, rho))


def test_left_and_right_duals_differ():
    # without a pivotal structure the left dual (S^-1) differs from the
    # right dual (S) whenever S^2 != id, as for Sweedler's H4
    H4 = HopfAlgebra.sweedler()
    W = Representation[H4].regular()
    assert W.l.is_module() and W.r.is_module()
    left = W.l.action.eval(dtype=complex).array
    right = W.r.action.eval(dtype=complex).array
    assert not np.allclose(left, right)
    # for a group algebra S^2 = id, so the two duals coincide
    Z2 = HopfAlgebra.cyclic(2)
    U = Representation[Z2].regular()
    assert np.allclose(U.l.action.eval(dtype=complex).array,
                       U.r.action.eval(dtype=complex).array)


def test_functor_maps_winding_to_dual():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    assert F(x).action == V.action
    assert F(x.r).action == V.r.action != V.action


def test_twist_from_braid():
    # the twist is the trace of the braid, defined without a ribbon element
    D = HopfAlgebra.cyclic(2).double()

    def theta(rep):
        return complex(
            Intertwiner[D].twist(rep).eval(dtype=complex).array)

    Vem = Representation[D].direct_sum(
        [Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)])
    assert np.allclose(
        Intertwiner[D].twist(Vem).eval(dtype=complex).array, np.eye(2))
    assert np.isclose(theta(Representation[D].anyon(1, -1)), -1)  # fermion
    assert np.isclose(theta(Representation[D].anyon(0, 1)), 1)    # vacuum


def test_is_module_rejects_non_module():
    Z2 = HopfAlgebra.cyclic(2)
    ty = Z2.ty
    zero = Box('bad', ty @ Dim(2), Dim(2), np.zeros((2, 2, 2)))
    assert not Representation[Z2](action=zero).is_module()   # rho(1) != id


def test_snake_equations():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    identity = F(ribbon.Id(x)).eval(dtype=complex).array
    assert np.allclose(F(left).eval(dtype=complex).array, identity)
    assert np.allclose(F(right).eval(dtype=complex).array, identity)


# -- the functor and the topological invariant -------------------------------

def test_functor_returns_a_tensor_network():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    network = Functor(
        ob_map={x: V}, ar_map={}, cod=Intertwiner[D])(_hopf_link(x))
    assert isinstance(network, tensor.Diagram)      # not a contracted Tensor
    # the user contracts it themselves with .eval
    assert np.isclose(complex(network.eval(dtype=complex)), 0)


def test_reidemeister_moves():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
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
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    circle = complex(F(_circle(x)).eval(dtype=complex))
    unlink = complex(F(_unlink(x)).eval(dtype=complex))
    hopf = complex(F(_hopf_link(x)).eval(dtype=complex))
    assert np.isclose(circle, 2)       # unknot -> qdim
    assert np.isclose(unlink, 4)       # 2 unknots
    assert np.isclose(hopf, 0)         # Hopf link
    assert not np.isclose(hopf, unlink)


def test_crossing_number_distinguishes_closures():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
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
    e = Representation[D].anyon(0, -1)
    m = Representation[D].anyon(1, 1)
    xe, xm = ribbon.Ty('e'), ribbon.Ty('m')
    F = Functor(ob_map={xe: e, xm: m}, ar_map={}, cod=Intertwiner[D])

    def value(a, b):
        link = (ribbon.Braid(a, b) >> ribbon.Braid(b, a)).trace(n=2)
        return complex(F(link).eval(dtype=complex))

    assert np.isclose(value(xe, xm), -1)   # mutual statistics -1
    assert np.isclose(value(xe, xe), 1)
    assert np.isclose(value(xm, xm), 1)


def test_functor_on_generic_box():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    box = ribbon.Box('f', x, x)
    F = Functor(ob_map={x: V}, ar_map={box: np.array([[1, 2], [3, 4]])},
                cod=Intertwiner[D])
    assert np.allclose(
        F(box).eval(dtype=complex).array, [[1, 2], [3, 4]])


def test_functor_on_type():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    assert F(x) == Dim(2)
    assert F(x @ x) == Dim(2, 2)


def test_twist_in_diagram():
    # a group algebra: trivial twist, but exercises the Twist functor path
    Z2 = HopfAlgebra.cyclic(2)
    V = Representation[Z2].regular()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[Z2])
    twist = ribbon.Twist(x)
    assert np.allclose(
        F(twist).eval(dtype=complex).array, np.eye(2))
    assert np.allclose(
        F(twist.dagger()).eval(dtype=complex).array, np.eye(2))
