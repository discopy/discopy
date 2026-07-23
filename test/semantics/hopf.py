import numpy as np

from discopy import ribbon, tensor
from discopy.tensor import Dim, Box
from discopy.hopf import (
    HopfAlgebra, Double, Representation, Intertwiner, Functor)


def _circle(x):
    return ribbon.Cap(x, x.r) >> ribbon.Cup(x, x.r)


def _unlink(x):
    return _circle(x) @ _circle(x)


def _hopf_link(x):
    braid = ribbon.Braid(x, x)
    return (braid >> braid).trace(n=2)


def _double_and_module():
    D = Double(HopfAlgebra.cyclic(2))
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    return D, Representation[D].direct_sum([e, m])


def test_group_algebra_is_valid():
    for n in [1, 2, 3, 5]:
        assert HopfAlgebra.cyclic(n).is_valid()
    table = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    assert HopfAlgebra.group_algebra(table).is_valid()


def test_double_is_quasitriangular_hopf_algebra():
    for n in [2, 3]:
        D = Double(HopfAlgebra.cyclic(n))
        assert D.dim == n * n
        assert D.is_valid()
        assert D.is_quasitriangular()


def test_commutativity_properties():
    """D(Z/2) is commutative and cocommutative, Sweedler's H4 is neither."""
    D = Double(HopfAlgebra.cyclic(2))
    assert D.is_commutative() and D.is_cocommutative()
    H4 = HopfAlgebra.sweedler()
    assert not H4.is_commutative() and not H4.is_cocommutative()


def test_double_of_sweedler():
    """The double of H4 exercises S^-1 in the coadjoint multiplication."""
    H4 = HopfAlgebra.sweedler()
    assert H4.is_valid() and H4.dim == 4
    Sarr = H4.antipode.eval(dtype=complex).array
    assert not np.allclose(Sarr @ Sarr, np.eye(4))
    D = Double(H4)
    assert D.dim == 16
    assert D.is_valid() and D.is_quasitriangular()


def test_double_antipode_inverse():
    """S^-1 = u^-1 S(x) u for the Drinfeld element u = S(R'')R', with
    u^-1 = R''S^2(R'), also when S^2 != id. The composite is too deep for a
    single einsum (issue #447), so assemble it from the generator matrices."""
    for base in [HopfAlgebra.cyclic(2), HopfAlgebra.sweedler()]:
        D, n = Double(base), base.dim ** 2
        S = D.antipode.eval(dtype=complex).array.reshape(n, n)
        R = D.R.eval(dtype=complex).array.reshape(n, n)
        mult = D.mult.eval(dtype=complex).array.reshape(n, n, n)
        u = np.einsum('ij,ja,aik->k', R, S, mult)
        u_inv = np.einsum('ij,ib,ba,jak->k', R, S, S, mult)
        Si = np.einsum('xy,i,iyk,j,kjl->xl', S, u_inv, mult, u, mult)
        assert np.allclose(Si @ S, np.eye(n))
        assert np.allclose(S @ Si, np.eye(n))


def test_double_is_fine_grained():
    """Every box in the double's generators acts on the base object, so no
    structure tensor of the doubled dimension is ever materialised."""
    D = Double(HopfAlgebra.cyclic(2))
    for gen in D.generators:
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
    try:
        Intertwiner[noR].braid(W, W)
        assert False
    except ValueError:
        pass
    try:
        Intertwiner.braid(W, W)
        assert False
    except ValueError:
        pass
    assert W.l.is_module()


def test_non_invertible_antipode_is_flagged():
    Z2 = HopfAlgebra.cyclic(2)
    zero = Box('S', Z2.ty, Z2.ty, np.zeros((2, 2)))
    H = HopfAlgebra(Z2.unit, Z2.counit, Z2.mult, Z2.comult, zero)
    try:
        H.antipode_inv
        assert False
    except ValueError:
        pass


def test_class_generic_over_the_algebra():
    """The algebra lives on the class, parametrisation caches by equality."""
    D = Double(HopfAlgebra.cyclic(2))
    assert Representation[D] is Representation[Double(HopfAlgebra.cyclic(2))]
    assert Representation[D].algebra == D
    V = Representation[D].regular()
    assert V.algebra == D
    braid = Intertwiner[D].braid(V, V)
    assert Intertwiner[D].algebra == D and braid.algebra == D


def test_representation_is_a_dim():
    D, V = _double_and_module()
    assert isinstance(V, Dim) and V == Dim(2)
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert e.action != m.action
    assert e != Dim(2)
    trivial = Representation[D](Dim(2))
    assert trivial == Dim(2) and trivial.is_module()
    assert hash(trivial) == hash(Dim(2))
    from discopy import hopf, rigid, tensor  # noqa: F401  (used by eval)
    assert eval(repr(trivial)) == trivial


def test_representation_init_checks():
    D, V = _double_and_module()
    try:
        Representation(Dim(2))
        assert False
    except ValueError:
        pass
    try:
        Representation[D](Dim(2), action=D.counit)
        assert False
    except ValueError:
        pass
    try:
        Representation[D](2)
        assert False
    except TypeError:
        pass


def test_anyon_needs_a_double():
    Z2 = HopfAlgebra.cyclic(2)
    try:
        Representation[Z2].anyon(0, 1)
        assert False
    except ValueError:
        pass


def test_repr_is_transparent():
    from discopy import hopf, tensor  # noqa: F401  (used by eval)
    H = HopfAlgebra.cyclic(2)
    assert eval(repr(H)) == H
    D = Double(H)
    assert eval(repr(D)) == D
    V = Representation[H].regular()
    assert eval(repr(V)) == V and eval(repr(V)).action == V.action


def test_representation_is_module():
    D, V = _double_and_module()
    assert V.is_module() and V == Dim(2)
    assert Representation[D].regular().is_module()
    for anyon in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        assert Representation[D].anyon(*anyon).is_module()


def test_tensor_of_representations():
    """Products act through the comultiplication, plain Dims trivially."""
    D, V = _double_and_module()
    e, m = Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)
    assert (e @ m).is_module() and (e @ m).action is not None
    VV = V @ V
    assert VV == Dim(2, 2) and VV.is_module()
    assert (V @ e @ m).is_module()
    assert (V @ Dim(2)).is_module() and (V @ Dim(2)) == Dim(2, 2)
    unit = Representation[D]()
    assert unit @ unit == Dim(1) and (unit @ V).action == V.action


def test_braiding_yang_baxter_and_inverse():
    D, V = _double_and_module()
    d = 2
    c = Intertwiner[D].braid(V, V).eval(dtype=complex).array
    c = c.reshape(d * d, d * d)
    assert not np.isclose(np.linalg.det(c), 0)
    swap = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            swap[a * d + b, b * d + a] = 1
    assert not np.allclose(c, swap)
    ci = Intertwiner[D].braid(V, V, is_dagger=True)\
        .eval(dtype=complex).array
    ci = ci.reshape(d * d, d * d)
    assert np.allclose(c.T @ ci.T, np.eye(d * d))
    R, eye = c.T, np.eye(d)
    R12, R23 = np.kron(R, eye), np.kron(eye, R)
    assert np.allclose(R12 @ R23 @ R12, R23 @ R12 @ R23)


def test_quantum_dimension():
    _, V = _double_and_module()
    assert np.isclose(V.qdim(), 2)


def test_dual_representation():
    """The right dual carries the twisted action rho(S h)^T, also when
    S^2 != id and on structured types."""
    D, V = _double_and_module()
    assert V.r.is_module()
    assert Representation[HopfAlgebra.sweedler()].regular().r.is_module()
    n = D.dim
    rho = V.action.eval(dtype=complex).array.reshape(n, 2, 2)
    S = D.antipode.eval(dtype=complex).array.reshape(n, n)
    dual = V.r.action.eval(dtype=complex).array.reshape(n, 2, 2)
    assert np.allclose(dual, np.einsum('hk,kio->hoi', S, rho))


def test_left_and_right_duals_differ():
    """The left dual twists by S^-1, so it differs from the right dual
    whenever S^2 != id, as for Sweedler's H4 but not a group algebra."""
    H4 = HopfAlgebra.sweedler()
    W = Representation[H4].regular()
    assert W.l.is_module() and W.r.is_module()
    left = W.l.action.eval(dtype=complex).array
    right = W.r.action.eval(dtype=complex).array
    assert not np.allclose(left, right)
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
    """The twist is the trace of the braid: the identity on e (+) m, minus
    one on the fermion, one on the vacuum."""
    D = Double(HopfAlgebra.cyclic(2))

    def theta(rep):
        return complex(
            Intertwiner[D].twist(rep).eval(dtype=complex).array)

    Vem = Representation[D].direct_sum(
        [Representation[D].anyon(0, -1), Representation[D].anyon(1, 1)])
    assert np.allclose(
        Intertwiner[D].twist(Vem).eval(dtype=complex).array, np.eye(2))
    assert np.isclose(theta(Representation[D].anyon(1, -1)), -1)
    assert np.isclose(theta(Representation[D].anyon(0, 1)), 1)


def test_is_module_rejects_non_module():
    Z2 = HopfAlgebra.cyclic(2)
    zero = Box('bad', Z2.ty @ Dim(2), Dim(2), np.zeros((2, 2, 2)))
    assert not Representation[Z2](Dim(2), zero).is_module()


def test_snake_equations():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    left = ribbon.Id(x.l).transpose(left=True)
    right = ribbon.Id(x.r).transpose(left=False)
    identity = F(ribbon.Id(x)).eval(dtype=complex).array
    assert np.allclose(F(left).eval(dtype=complex).array, identity)
    assert np.allclose(F(right).eval(dtype=complex).array, identity)


def test_functor_returns_a_tensor_network():
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    network = Functor(
        ob_map={x: V}, ar_map={}, cod=Intertwiner[D])(_hopf_link(x))
    assert isinstance(network, tensor.Diagram)
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
    assert np.isclose(circle, 2)
    assert np.isclose(unlink, 4)
    assert np.isclose(hopf, 0)
    assert not np.isclose(hopf, unlink)


def test_crossing_number_distinguishes_closures():
    """The unlink, the unknot and the Hopf link as braid closures."""
    D, V = _double_and_module()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[D])
    X = ribbon.Ty('x')
    closures = [
        ribbon.Id(X @ X).trace(n=2),
        ribbon.Braid(X, X).trace(n=2),
        (ribbon.Braid(X, X) >> ribbon.Braid(X, X)).trace(n=2),
    ]
    values = [complex(F(c).eval(dtype=complex)) for c in closures]
    assert np.allclose(values, [4, 2, 0])


def test_two_colour_mutual_braiding():
    """The e and m anyons of the toric code have mutual statistics -1."""
    D = Double(HopfAlgebra.cyclic(2))
    e = Representation[D].anyon(0, -1)
    m = Representation[D].anyon(1, 1)
    xe, xm = ribbon.Ty('e'), ribbon.Ty('m')
    F = Functor(ob_map={xe: e, xm: m}, ar_map={}, cod=Intertwiner[D])

    def value(a, b):
        link = (ribbon.Braid(a, b) >> ribbon.Braid(b, a)).trace(n=2)
        return complex(F(link).eval(dtype=complex))

    assert np.isclose(value(xe, xm), -1)
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
    """A group algebra has a trivial twist, mapped through the functor."""
    Z2 = HopfAlgebra.cyclic(2)
    V = Representation[Z2].regular()
    x = ribbon.Ty('x')
    F = Functor(ob_map={x: V}, ar_map={}, cod=Intertwiner[Z2])
    twist = ribbon.Twist(x)
    assert np.allclose(
        F(twist).eval(dtype=complex).array, np.eye(2))
    assert np.allclose(
        F(twist.dagger()).eval(dtype=complex).array, np.eye(2))
