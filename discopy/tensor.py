# -*- coding: utf-8 -*-

"""
The category of matrices with the Kronecker product as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Dim
    Tensor
    Functor
    Diagram
    CMap
    Box
    Swap
    Cup
    Cap
    Spider
    Sum
    Bubble
"""

from __future__ import annotations

from itertools import count
from typing import Callable, TYPE_CHECKING

from discopy import (
    cat, monoidal, rigid, symmetric, frobenius)
from discopy.cat import ar_factory, assert_iscomposable
from discopy.frobenius import Dim, Cup
from discopy.matrix import (  # noqa: F401
    Matrix, backend, set_backend, get_backend, NumPy, JAX)
from discopy.abc import NamedGeneric
from discopy.utils import (
    factory_name, assert_isinstance, product, assert_isatomic)

if TYPE_CHECKING:
    import sympy
    import tensornetwork
    import quimb


@ar_factory
class Tensor(Matrix):
    """
    A tensor is a :class:`Matrix` with dimensions as domain and codomain and
    the Kronecker product as tensor.

    Parameters:
        inside : The array inside the tensor.
        dom : The domain dimension.
        cod : The codomain dimension.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            dagger
            cups
            caps
            swap
            spiders
            transpose
            conjugate
            round
            subs
            grad
            jacobian

    Examples
    --------
    >>> m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    >>> v = Tensor([0, 1], Dim(1), Dim(2))
    >>> v >> m >> v.dagger()
    Tensor[int64]([0], dom=Dim(1), cod=Dim(1))

    Notes
    -----
    Tensors can have sympy symbols as free variables.

    >>> from sympy import Expr
    >>> from sympy.abc import phi, psi
    >>> v = Tensor[Expr]([phi, psi], Dim(1), Dim(2))
    >>> d = v >> v.dagger()
    >>> assert v >> v.dagger() == Tensor[Expr](
    ...     [phi * phi.conjugate() + psi * psi.conjugate()], Dim(1), Dim(1))

    These can be substituted and lambdifed.

    >>> v.subs(phi, 0).lambdify(psi, dtype=int)(1)
    Tensor[int]([0, 1], dom=Dim(1), cod=Dim(2))

    We can also use jax.numpy using :func:`backend`.

    >>> with backend('jax'):
    ...     f = lambda *xs: d.lambdify(phi, psi, dtype=float)(*xs).array
    ...     import jax
    ...     assert jax.grad(f)(1., 2.) == 2.
    """
    ob = Dim

    def __init__(self, array, dom: Dim, cod: Dim):
        assert_isinstance(dom, Dim)
        assert_isinstance(cod, Dim)
        super().__init__(array, product(dom.inside), product(cod.inside))
        self.array = self.array.reshape(dom.inside + cod.inside)
        self.dom, self.cod = dom, cod

    @classmethod
    def id(cls, dom=Dim(1)) -> Tensor:
        return cls(Matrix.id(product(dom.inside)).array, dom, dom)

    def then(self, other: Tensor = None, *others: Tensor) -> Tensor:
        if other is None or others:
            return super().then(other, *others)
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        with backend() as np:
            array = np.tensordot(self.array, other.array, len(self.cod))\
                if self.array.shape and other.array.shape\
                else self.array * other.array
        return type(self)(array, self.dom, other.cod)

    def tensor(self, other: Tensor = None, *others: Tensor) -> Tensor:
        if other is None or others:
            return Diagram.tensor(self, other, *others)
        assert_isinstance(other, Tensor)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        source = range(len(dom @ cod))
        target = [
            i if i < len(self.dom) or i >= len(self.dom @ other.dom @ self.cod)
            else i - len(self.cod) if i >= len(self.dom @ self.cod)
            else i + len(other.dom) for i in source]
        with backend() as np:
            array = np.tensordot(self.array, other.array, 0)\
                if self.array.shape and other.array.shape\
                else self.array * other.array
            array = np.moveaxis(array, source, target)
        return type(self)(array, dom, cod)

    def dagger(self) -> Tensor:
        source = range(len(self.dom @ self.cod))
        target = [i + len(self.cod) if i < len(self.dom) else
                  i - len(self.dom) for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, list(source), target))
        return type(self)(array, self.cod, self.dom)

    @classmethod
    def cup_factory(cls, left: Dim, right: Dim) -> Tensor:
        assert_isinstance(left, Dim)
        assert_isinstance(right, Dim)
        left.assert_isadjoint(right)
        return cls(cls.id(left).array, left @ right, Dim(1))

    @classmethod
    def cups(cls, left: Dim, right: Dim) -> Tensor:
        return rigid.nesting(cls, cls.cup_factory)(left, right)

    @classmethod
    def caps(cls, left: Dim, right: Dim) -> Tensor:
        return cls.cups(left, right).dagger()

    @classmethod
    def swap(cls, left: Dim, right: Dim) -> Tensor:
        dom, cod = left @ right, right @ left
        array = cls.id(dom).array
        source = range(len(dom), 2 * len(dom))
        target = [i + len(right) if i < len(dom @ left)
                  else i - len(left) for i in source]
        with backend() as np:
            return cls(np.moveaxis(array, source, target), dom, cod)

    @classmethod
    def spider_factory(cls, n_legs_in: int, n_legs_out: int,
                       typ: Dim, phase=None) -> Tensor:
        if phase is not None:
            raise NotImplementedError
        assert_isatomic(typ, Dim)
        n, = typ.inside
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        with backend('numpy'):
            result = cls.zero(dom, cod)
            for i in range(n):
                result.array[len(dom @ cod) * (i, )] = 1
            return result

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Dim, phase=None
                ) -> Tensor:
        """
        The tensor of interleaving spiders.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return frobenius.Diagram.spiders.__func__(
            cls, n_legs_in, n_legs_out, typ, phase)

    @classmethod
    def copy(cls, x: Dim, n: int) -> Tensor:
        """
        Constructs spiders of dimension `x` with one leg in and `n` legs out.

        Parameters:
            x : The type of the spiders.
            n : The number of legs out for each spider.

        Example
        -------
        >>> from discopy import markov
        >>> n = markov.Ty('n')
        >>> F = Functor(ob={n: Dim(2)}, ar={}, dom=markov.Diagram)
        >>> assert F(markov.Copy(n, 2)) == Tensor[int].copy(Dim(2), 2)\\
        ...     == Tensor[int]([1, 0, 0, 0, 0, 0, 0, 1], Dim(2), Dim(2, 2))
        """
        return cls.spiders(1, n, x)

    def transpose(self, left=False) -> Tensor:
        """
        Returns the diagrammatic transpose.

        Note
        ----
        This is *not* the same as the algebraic transpose for non-atomic dims.
        """
        return type(self)(
            self.array.transpose(), self.cod[::-1], self.dom[::-1])

    l = r = property(transpose)

    def conjugate(self, diagrammatic=True) -> Tensor:
        """
        Returns the conjugate of a tensor.

        Parameters
        ----------
        diagrammatic : bool, default: True
            Whether to use the diagrammatic or algebraic conjugate.
        """
        if not diagrammatic:
            with backend() as np:
                return Tensor[self.dtype](
                    np.conj(self.array), self.dom, self.cod)
        # reverse the wires for both inputs and outputs
        source = range(len(self.dom @ self.cod))
        target = [
            len(self.dom) - i - 1 for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, list(source), target))
        return type(self)(array, self.dom[::-1], self.cod[::-1])

    @classmethod
    def zero(cls, dom: Dim, cod: Dim) -> Tensor:
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zero(Dim(2), Dim(2))\\
        ...     == Tensor([0, 0, 0, 0], Dim(2), Dim(2))
        """
        with backend() as np:
            return cls(np.zeros((dom @ cod).inside, dtype=cls.dtype or int),
                       dom, cod)

    def jacobian(self, *variables: "list[sympy.Symbol]", **params) -> Tensor:
        """
        Jacobian with respect to :code:`variables`.

        Parameters:
            variables : The list of variables to differentiate.

        Returns
        -------
        tensor : Tensor
            with :code:`tensor.dom == self.dom`
            and :code:`tensor.cod == Dim(len(variables)) @ self.cod`.

        Examples
        --------
        >>> from sympy import Expr
        >>> from sympy.abc import x, y, z
        >>> vector = Tensor[Expr]([x ** 2, y * z], Dim(1), Dim(2))
        >>> vector.jacobian(x, y, z)
        Tensor[Expr]([2*x, 0, 0, z, 0, y], dom=Dim(1), cod=Dim(3, 2))
        """
        dim = Dim(len(variables) or 1)
        result = self.zero(self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = self.zero(Dim(1), dim)
            onehot.array[i] = 1
            result += onehot @ self.grad(var)
        return result


class Functor(frobenius.Functor):
    """
    A tensor functor is a frobenius functor with a domain category ``dom``
    and ``Tensor[dtype]`` as codomain for a given ``dtype``.

    Parameters:
        ob : The object mapping.
        ar : The arrow mapping.
        dom : The domain of the functor.
        dtype : The datatype for the codomain ``Tensor[dtype]``.
        contractor : The contraction strategy for diagrams: ``None`` for the
            naive functor, ``'tn'``, ``'quimb'`` or ``'einsum'``, see
            :attr:`CONTRACTORS`.
        backend : The name of the array :func:`backend` to evaluate in,
            default is the active backend.

    Example
    -------
    >>> n, s = map(rigid.Ty, "ns")
    >>> Alice = rigid.Box('Alice', rigid.Ty(), n)
    >>> loves = rigid.Box('loves', rigid.Ty(), n.r @ s @ n.l)
    >>> Bob = rigid.Box('Bob', rigid.Ty(), n)
    >>> diagram = Alice @ loves @ Bob\\
    ...     >> rigid.Cup(n, n.r) @ s @ rigid.Cup(n.l, n)

    >>> F = Functor(
    ...     ob={s: 1, n: 2},
    ...     ar={Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]},
    ...     dom=rigid.Diagram, dtype=bool)
    >>> F(diagram)
    Tensor[bool]([True], dom=Dim(1), cod=Dim(1))

    The same functor with any choice of contractor gives the same tensor.

    >>> for contractor in CONTRACTORS:
    ...     F_ = Functor(F.ob_map, F.ar_map, dom=rigid.Diagram,
    ...                  dtype=bool, contractor=contractor)
    ...     assert F_(diagram) == F(diagram)

    >>> rewrite = diagram\\
    ...     .transpose_box(2).transpose_box(0, left=True).normal_form()
    >>> from discopy.drawing import Equation
    >>> Equation(diagram, rewrite).draw(
    ...     figsize=(8, 3), path='docs/_static/tensor/rewrite.png')

    .. image :: /_static/tensor/rewrite.png
        :align: center

    >>> assert F(diagram) == F(rewrite)
    """
    dom, cod = frobenius.Diagram, Tensor

    def __init__(
            self, ob: dict[cat.Ob, Dim], ar: dict[cat.Box, list],
            dom: type = None, dtype: type = int,
            contractor: str = None, backend: str = None):
        self.dtype = dtype
        self.contractor, self.backend = contractor, backend
        cod = type(self).cod[dtype]
        super().__init__(ob, ar, dom=dom or type(self).dom, cod=cod)

    def __repr__(self):
        contractor = "" if self.contractor is None\
            else f", contractor={self.contractor!r}"
        backend = "" if self.backend is None\
            else f", backend={self.backend!r}"
        return factory_name(type(self))\
            + f"(ob_map={self.ob_map}, ar_map={self.ar_map}, "\
            + f"dom={factory_name(self.dom)}, "\
            + f"dtype={self.dtype.__name__}{contractor}{backend})"

    def contract(self, other: monoidal.Diagram) -> Tensor:
        """
        Evaluate a diagram as a tensor network with the chosen contractor,
        by mapping it to a :class:`Diagram` with tensor boxes.

        Parameters:
            other : The diagram to evaluate.
        """
        dtype = self.dtype
        diagram = frobenius.Functor(
            ob=self.ob_map, ar=lambda box: Box[dtype](
                box.name, self(box.dom), self(box.cod), self.ar_map[box]),
            dom=self.dom, cod=Diagram)(other)
        if self.contractor is None:
            return diagram.eval(dtype=dtype)
        return CONTRACTORS[self.contractor](diagram, dtype)

    def __call__(self, other):
        if isinstance(other, Dim):
            return other
        if isinstance(other, Bubble):
            return self(other.arg).map(other.func)
        if isinstance(other, (cat.Ob, cat.Box)):
            return super().__call__(other)
        assert_isinstance(other, monoidal.Diagram)
        if self.contractor is not None or self.backend is not None:
            with backend(self.backend):
                return self.contract(other)
        dim = lambda scan: len(self(scan))
        scan, array = other.dom, Tensor.id(self(other.dom)).array
        for box, off in zip(other.boxes, other.offsets):
            if isinstance(box, symmetric.Swap):
                source = range(
                    dim(other.dom @ scan[:off]),
                    dim(other.dom @ scan[:off] @ box.dom))
                target = [
                    i + dim(box.right)
                    if i < dim(other.dom @ scan[:off]) + dim(box.left)
                    else i - dim(box.left) for i in source]
                with backend() as np:
                    array = np.moveaxis(array, list(source), list(target))
                scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
                continue
            left = dim(scan[:off])
            source = list(range(dim(other.dom) + left,
                                dim(other.dom) + left + dim(box.dom)))
            target = list(range(dim(box.dom)))
            with backend() as np:
                array = np.tensordot(array, self(box).array, (source, target))
            source = range(len(array.shape) - dim(box.cod), len(array.shape))
            target = range(dim(other.dom) + left,
                           dim(other.dom) + left + dim(box.cod))
            with backend() as np:
                array = np.moveaxis(array, list(source), list(target))
            scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
        return self.cod(array, self(other.dom), self(other.cod))


@ar_factory
class Diagram(NamedGeneric['dtype'], frobenius.Diagram):
    """
    A tensor diagram is a frobenius diagram with tensor boxes.

    Example
    -------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> diagram = vector[::-1] >> vector @ vector
    >>> print(diagram)
    vector[::-1] >> vector >> Dim(2) @ vector
    """
    ob = Dim

    def eval(self, contractor: Callable = None, dtype: type = None) -> Tensor:
        """
        Evaluate a tensor diagram as a :class:`Tensor`.

        Parameters:
            contractor : Use ``tensornetwork`` or :class:`Functor` by default.
            dtype : Used for spiders.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).eval().array == 1
        >>> from tensornetwork.contractors import auto
        >>> assert (vector >> vector[::-1]).eval(auto).array == 1
        """
        dtype = dtype or self.dtype
        if contractor is None:
            return Functor(
                ob=lambda x: x, ar=lambda f: f.array, dtype=dtype)(self)
        array = contractor(*self.to_tn(dtype=dtype)).tensor
        return Tensor[dtype](array, self.dom, self.cod)

    def to_quimb(self, dtype: type = None) -> "quimb.tensor.Tensor":
        """
        Convert a tensor diagram to a quimb tensor.

        Parameters:
            dtype : Used for spiders.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> t_net = (vector >> vector[::-1]).to_quimb()
        >>> assert t_net.contract(preserve_tensor=True).data == 1
        """
        import quimb.tensor as qtn
        inputs = [
                qtn.COPY_tensor(
                    d=getattr(dim, 'dim', dim),
                    inds=(f'inp{i}', f'inp{i}_end')
                ) for i, dim in enumerate(self.dom.inside)]
        tensors = inputs[:]
        scan = [(t, 1) for t in inputs]

        for i, (box, off) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box, Swap):
                scan[off], scan[off + 1] = scan[off + 1], scan[off]
                continue

            in_inds = [f't{i}_i{j}' for j in range(len(box.dom))]
            out_inds = [f't{i}_o{j}' for j in range(len(box.cod))]
            t = qtn.Tensor(
                data=box.eval().array,
                inds=in_inds + out_inds,
            )
            tensors.append(t)
            for j in range(len(box.dom)):
                other_t, other_ind = scan[off + j]
                qtn.connect(other_t, t, other_ind, j)

            scan[off:off + len(box.dom)] = [
                (t, len(box.dom) + ind) for ind in range(len(out_inds))
            ]

        for i, (t, j) in enumerate(scan):
            output = qtn.COPY_tensor(
                d=t.data.shape[j],
                inds=(f'out{i}_start', f'out{i}')
            )
            qtn.connect(t, output, j, 0)
            tensors.append(output)

        tensor_net = qtn.TensorNetwork(tensors)
        return tensor_net

    def to_tn(self, dtype: type = None) -> tuple[
            list["tensornetwork.Node"], list["tensornetwork.Edge"]]:
        """
        Convert a tensor diagram to :code:`tensornetwork`.

        Parameters:
            dtype : Used for spiders.

        Examples
        --------
        >>> import numpy as np
        >>> from tensornetwork import Node, Edge
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> nodes, output_edge_order = vector.to_tn()
        >>> node, = nodes
        >>> assert node.name == "vector" and np.all(node.tensor == [0, 1])
        >>> assert output_edge_order == [node[0]]
        """
        import tensornetwork as tn
        if dtype is None:
            dtype = self.dtype
        nodes = [
            tn.CopyNode(2, getattr(dim, 'dim', dim), f'input_{i}', dtype=dtype)
            for i, dim in enumerate(self.dom.inside)]
        inputs, outputs = [n[0] for n in nodes], [n[1] for n in nodes]
        for box, offset in zip(self.boxes, self.offsets):
            if isinstance(box, Swap):
                outputs[offset], outputs[offset + 1]\
                    = outputs[offset + 1], outputs[offset]
                continue
            if isinstance(box, (Cup, Spider)):
                dims = (len(box.dom), len(box.cod))
                if dims == (1, 1):  # identity
                    continue
                elif dims == (2, 0):  # cup
                    tn.connect(*outputs[offset:offset + 2])
                    del outputs[offset:offset + 2]
                    continue
                else:
                    node = tn.CopyNode(
                        sum(dims), outputs[offset].dimension, dtype=dtype)
            else:
                array = box.eval(dtype=dtype).array
                node = tn.Node(array, str(box))
            for i, _ in enumerate(box.dom):
                tn.connect(outputs[offset + i], node[i])
            outputs[offset:offset + len(box.dom)] = node[len(box.dom):]
            nodes.append(node)
        return nodes, inputs + outputs

    def grad(self, var, **params):
        """ Gradient with respect to :code:`var`. """
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        left, box, right, tail = tuple(self.inside[0]) + (self[1:], )
        t1 = self.id(left) @ box.grad(var, **params) @ self.id(right) >> tail
        t2 = self.id(left) @ box @ self.id(right) >> tail.grad(var, **params)
        return t1 + t2

    def jacobian(self, variables, **params) -> Diagram:
        """
        Diagrammatic jacobian with respect to :code:`variables`.

        Parameters
        ----------
        variables : List[sympy.Symbol]
            Differentiated variables.

        Returns
        -------
        tensor : Tensor
            with :code:`tensor.dom == self.dom`
            and :code:`tensor.cod == Dim(len(variables)) @ self.cod`.

        Examples
        --------
        >>> from sympy import Expr
        >>> from sympy.abc import x, y, z
        >>> vector = Box("v", Dim(1), Dim(2), [x ** 2, y * z])
        >>> vector.jacobian([x, y, z]).eval(dtype=Expr)
        Tensor[Expr]([2*x, 0, 0, z, 0, y], dom=Dim(1), cod=Dim(3, 2))
        """
        dim = Dim(len(variables) or 1)
        result = Sum((), self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = Tensor.zero(Dim(1), dim)
            onehot.array[i] = 1
            result += Box(str(var), Dim(1), dim, onehot.array) @ self.grad(var)
        return result


class CMap(frobenius.CMap):
    """
    A tensor combinatorial map is Einstein notation for a tensor network:
    boxes are tensors, the 2-cycles of the ``edges`` involution are the
    summed indices and the boundary ports are the free indices.

    Swaps, cups and caps become wiring while spiders stay as boxes, so that
    every wire has exactly two ends and the whole map contracts in a single
    ``einsum`` call.

    Example
    -------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> assert (vector >> vector[::-1]).to_map().eval().array == 1
    """
    category = Diagram

    def to_einsum(self, dtype: type = None) -> tuple[
            list, list[list[int]], list[int]]:
        """
        The operands ``arrays``, their integer axis labels ``indices`` and
        the free ``output`` labels of the map, for the interleaved form
        ``np.einsum(arrays[0], indices[0], ..., output)`` under the active
        :func:`backend`. Integer labels lift the 52-index limit of subscript
        strings.

        Parameters:
            dtype : Used for spiders.

        Example
        -------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> arrays, indices, output = (vector >> vector[::-1])\\
        ...     .to_map().to_einsum()
        >>> with backend() as np:
        ...     operands = [x for pair in zip(arrays, indices) for x in pair]
        ...     assert np.einsum(*operands, output) == 1
        """
        wires, fresh = {}, count()
        for source, target in enumerate(self.edges):
            if source <= target:
                wires[source] = wires[target] = next(fresh)
        dim = lambda typ: product(typ.inside)
        ports = self.ports
        arrays, indices, output = [], [], []
        with backend() as np:
            for port in range(len(self.dom)):
                label = next(fresh)
                arrays.append(np.array(np.eye(dim(ports[port].obj))))
                indices.append([label, wires[port]])
                output.append(label)
            start = len(self.dom)
            for box in self.boxes:
                arity, coarity = len(box.dom), len(box.cod)
                box_ports = list(range(start, start + arity)) + list(
                    reversed(range(start + arity, start + arity + coarity)))
                arrays.append(box.eval(dtype=dtype).array)
                indices.append([wires[port] for port in box_ports])
                start += arity + coarity
            for port in range(self.n_ports - len(self.cod), self.n_ports):
                label = next(fresh)
                arrays.append(np.array(np.eye(dim(ports[port].obj))))
                indices.append([wires[port], label])
                output.append(label)
            for scalar in self.scalars:
                arrays.append(np.array(np.eye(dim(scalar))))
                indices.append(2 * [next(fresh)])
        return arrays, indices, output

    @staticmethod
    def einsum_params(operands: list, output: list[int],
                      optimize: str) -> dict:
        """
        The ``optimize`` parameter for ``np.einsum`` under the active
        :func:`backend`: an explicit ``opt_einsum`` path for numpy when
        available (numpy's own pathfinder chokes on large maps where
        ``opt_einsum``, a dependency of jax, does not), the given strategy
        for numpy and jax otherwise and nothing for other backends.
        """
        if not isinstance(get_backend(), (NumPy, JAX)):
            return {}
        if isinstance(get_backend(), NumPy):
            try:
                from opt_einsum import contract_path
                path, _ = contract_path(*operands, output, optimize=optimize)
                return dict(optimize=['einsum_path'] + list(path))
            except ImportError:
                pass
        return dict(optimize=optimize)

    def eval(self, dtype: type = None, optimize: str = "greedy") -> Tensor:
        """
        Contract the map as a single ``einsum`` under the active
        :func:`backend`, e.g. ``jax.numpy`` for autodiff.

        Parameters:
            dtype : Used for spiders.
            optimize : The contraction path strategy, see
                :meth:`einsum_params`. The default ``"greedy"`` avoids the
                exponential ``"optimal"`` search on large maps.

        Example
        -------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).to_map().eval().array == 1

        >>> with backend('jax'):
        ...     import jax, jax.numpy as jnp
        ...     b = lambda x: Box[float]('v', Dim(1), Dim(2), x * jnp.ones(2))
        ...     f = lambda x: (b(x) >> b(x)[::-1]).to_map().eval().array
        ...     assert jax.grad(f)(1.) == 4.
        """
        cls = Tensor if dtype is None else Tensor[dtype]
        arrays, indices, output = self.to_einsum(dtype=dtype)
        if not arrays:
            return cls([1], self.dom, self.cod)
        operands = [x for pair in zip(arrays, indices) for x in pair]
        with backend() as np:
            array = np.einsum(*operands, output, **self.einsum_params(
                operands, output, optimize))
        return cls(array, self.dom, self.cod)


def tn_contractor(diagram: Diagram, dtype: type) -> Tensor:
    """ Contract a tensor diagram with ``tensornetwork``. """
    from tensornetwork.contractors import auto
    return diagram.eval(auto, dtype=dtype)


def quimb_contractor(diagram: Diagram, dtype: type) -> Tensor:
    """ Contract a tensor diagram with ``quimb``. """
    output_inds = [f"inp{i}" for i, _ in enumerate(diagram.dom.inside)]\
        + [f"out{i}" for i, _ in enumerate(diagram.cod.inside)]
    array = diagram.to_quimb(dtype=dtype).contract(
        output_inds=output_inds, preserve_tensor=True).data
    return Tensor[dtype](array, diagram.dom, diagram.cod)


def einsum_contractor(diagram: Diagram, dtype: type) -> Tensor:
    """ Contract a tensor diagram as one ``einsum`` via :class:`CMap`. """
    return diagram.to_map().eval(dtype=dtype)


CONTRACTORS = {
    'tn': tn_contractor,
    'quimb': quimb_contractor,
    'einsum': einsum_contractor,
}


class Box(frobenius.Box, Diagram):
    """
    A tensor box is a frobenius box with an array as data.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. its input dimension.
        cod : The codomain of the box, i.e. its output dimension.
        data : The array inside the tensor box.

    Example
    -------
    >>> b1 = Box('sauce_0', Dim(1), Dim(2), data=[0.84193562, 0.91343221])
    >>> b1.eval()
    Tensor[float64]([0.84193562, 0.91343221], dom=Dim(1), cod=Dim(2))
    """

    def __setstate__(self, state):
        NamedGeneric.__setstate__(self, state)
        if "data" not in state and state.get("_array", None) is not None:
            state['data'] = state['_array']
            del state["_array"]
        super().__setstate__(state)
        if self.dtype is None and self.data is not None:
            self.data, self.dtype = self._get_data_dtype(self.data)
            self.__class__ = self.__class__[self.dtype]

    def __new__(
            cls, name=None, dom=None, cod=None, data=None, *args, **kwargs):
        if cls.dtype is not None or data is None:
            return object.__new__(cls)
        data, dtype = cls._get_data_dtype(data)
        return cls.__new__(
            cls[dtype],  name, dom, cod, data, *args, **kwargs)

    @staticmethod
    def _get_data_dtype(data):
        with backend() as np:
            data = np.array(data)
            # The dtype of an np.arrays is a class that contains a type
            # attribute that is the actual type. However, other backends
            # have different structures, so this is the easiest option:
            dtype = getattr(data.dtype, "type", data.dtype)
            return data, dtype

    @property
    def array(self):
        if self.data is not None:
            with backend() as np:
                return np.array(self.data).reshape(
                    self.dom.inside + self.cod.inside)

    def grad(self, var, **params):
        return self.bubble(
            func=lambda x: getattr(x, "diff", lambda _: 0)(var),
            drawing_name=f"$\\partial {var}$")


class Cup(frobenius.Cup, Box):
    """
    A tensor cup is a frobenius cup in a tensor diagram.

    Parameters:
        left (Dim) : The atomic type.
        right (Dim) : Its adjoint.
    """


class Cap(frobenius.Cap, Box):
    """
    A tensor cap is a frobenius cap in a tensor diagram.

    Parameters:
        left (Dim) : The atomic type.
        right (Dim) : Its adjoint.
    """


class Swap(frobenius.Swap, Box):
    """
    A tensor swap is a frobenius swap in a tensor diagram.

    Parameters:
        left (Dim) : The type on the top left and bottom right.
        right (Dim) : The type on the top right and bottom left.
    """


class Spider(frobenius.Spider, Box):
    """
    A tensor spider is a frobenius spider in a tensor diagram.

    Parameters:
        n_legs_in (int) : The number of legs in.
        n_legs_out (int) : The number of legs out.
        typ (Dim) : The dimension of the spider.
        data : The phase of the spider.

    Examples
    --------
    >>> vector = Box('vec', Dim(1), Dim(2), [0, 1])
    >>> spider = Spider(1, 2, Dim(2))
    >>> assert (vector >> spider).eval() == (vector @ vector).eval()
    >>> from discopy.drawing import Equation
    >>> Equation(vector >> spider, vector @ vector).draw(
    ...     path='docs/_static/tensor/frobenius-example.png', figsize=(3, 2))

    .. image:: /_static/tensor/frobenius-example.png
        :align: center
    """


class Sum(monoidal.Sum, Box):
    """
    A formal sum of tensor diagrams with the same domain and codomain.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Dim) : The domain of the formal sum.
        cod (Dim) : The codomain of the formal sum.
    """


class Bubble(monoidal.Bubble, Box):
    """
    Bubble in a tensor diagram, applies a function elementwise.

    Parameters
    ----------
    inside : tensor.Diagram
        The diagram inside the bubble.
    func : callable
        The function to apply, default is :code:`lambda x: int(not x)`.

    Examples
    --------

    >>> men = Box("men", Dim(1), Dim(2), [0, 1])
    >>> mortal = Box("mortal", Dim(2), Dim(1), [1, 1])
    >>> men_are_mortal = (men >> mortal.bubble()).bubble()
    >>> assert men_are_mortal.eval(dtype=bool)
    >>> men_are_mortal.draw(wire_labels=False,
    ...                     path='docs/_static/tensor/men-are-mortal.png')

    .. image:: /_static/tensor/men-are-mortal.png
        :align: center

    >>> from sympy import Expr
    >>> from sympy.abc import x
    >>> f = Box('f', Dim(2), Dim(2), [1, 0, 0, x])
    >>> g = Box('g', Dim(2), Dim(2), [-x, 0, 0, 1])
    >>> def grad(diagram, var):
    ...     return diagram.bubble(
    ...         func=lambda x: getattr(x, "diff", lambda _: 0)(var),
    ...         drawing_name=f"d${var}$" )
    >>> lhs = grad(f >> g, x)
    >>> rhs = (grad(f, x) >> g) + (f >> grad(g, x))
    >>> assert lhs.eval(dtype=Expr) == rhs.eval(dtype=Expr)

    >>> from discopy.drawing import Equation
    >>> Equation(lhs, rhs).draw(figsize=(5, 2), wire_labels=False,
    ...                         path='docs/_static/tensor/product-rule.png')

    .. image:: /_static/tensor/product-rule.png
        :align: center
    """

    def __init__(self, inside, func=lambda x: int(not x), **params):
        self.func = func
        super().__init__(inside, **params)

    def grad(self, var, **params):
        """
        The gradient of a bubble is given by the chain rule.

        >>> from sympy.abc import x
        >>> g = Box('g', Dim(2), Dim(2), [2 * x, 0, 0, x + 1])
        >>> f = lambda d: d.bubble(func=lambda x: x ** 2, drawing_name="f")
        >>> lhs, rhs = Box.grad(f(g), x), f(g).grad(x)

        >>> from discopy.drawing import Equation
        >>> Equation(lhs, rhs).draw(wire_labels=False,
        ...                         path='docs/_static/tensor/chain-rule.png')

        .. image:: /_static/tensor/chain-rule.png
            :align: center
        """
        from sympy import Symbol
        tmp = Symbol("tmp")
        name = "$\\frac{{\\partial {}}}{{\\partial {}}}$"
        return Spider(1, 2, self.dom)\
            >> self.arg.bubble(
                func=lambda x: self.func(tmp).diff(tmp).subs(tmp, x),
                drawing_name=name.format(self.drawing_name, var))\
            @ self.arg.grad(var) >> Spider(2, 1, self.cod)


Diagram.sum_factory, Diagram.braid_factory = Sum, Swap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.spider_factory, Diagram.bubble_factory = Spider, Bubble
Diagram.map_factory = CMap
Id = Diagram.id
