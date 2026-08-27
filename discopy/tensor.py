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

Tensor combinatorial maps
-------------------------

A :class:`CMap` is a tensor network stored as a combinatorial map, whose
boxes are tensors, edges are summed indices and boundary ports are free
indices. Swaps, cups and caps become wiring while spiders stay as boxes.

>>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
>>> assert (vector >> vector[::-1]).to_map().eval().array == 1

>>> with backend('jax'):  # doctest: +EXTRA
...     import jax, jax.numpy as jnp
...     b = lambda x: Box[float]('v', Dim(1), Dim(2), x * jnp.ones(2))
...     f = lambda x: (b(x) >> b(x)[::-1]).to_map().eval().array
...     assert jax.grad(f)(1.) == 4.
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Sequence

from discopy import (
    cat, monoidal, rigid, frobenius, cmap, config)
from discopy.cat import factory, assert_iscomposable
from discopy.frobenius import Dim, Cup
from discopy.matrix import (  # noqa: F401
    Matrix, backend, set_backend, get_backend,
    NumPy, JAX, PyTorch, TensorFlow)
from discopy.abc import NamedGeneric
from discopy.python import finset
from discopy.utils import (
    factory_name, assert_isinstance, product, assert_isatomic)

if TYPE_CHECKING:
    import sympy
    import tensornetwork
    import quimb


@factory
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

    >>> from sympy import Expr  # doctest: +EXTRA
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
        source = list(range(len(self.dom @ self.cod)))
        target = [i + len(self.cod) if i < len(self.dom) else
                  i - len(self.dom) for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, source, target))
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
        source = list(range(len(dom), 2 * len(dom)))
        target = [i + len(right) if i < len(dom @ left)
                  else i - len(left) for i in source]
        with backend() as np:
            return cls(np.moveaxis(array, source, target), dom, cod)

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[Dim]) -> Tensor:
        xs = finset.Permutation(xs, len(doms))
        dom = cls.ob.unit().tensor(*doms)
        if xs.is_identity:
            return cls.id(dom)
        offsets = [0]
        for dim in doms:
            offsets.append(offsets[-1] + len(dim))
        axes = finset.Permutation([
            axis for i in xs for axis in range(offsets[i], offsets[i + 1])])
        source = list(range(len(dom), 2 * len(dom)))
        target = [len(dom) + axis for axis in axes.dagger()]
        cod = Dim().tensor(*(doms[i] for i in xs))
        with backend() as np:
            array = np.moveaxis(cls.id(dom).array, source, target)
        return cls(array, dom, cod)

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
        if isinstance(get_backend(), NumPy):
            return result
        with backend() as np:
            return cls(np.array(result.array), dom, cod)

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
        >>> F = Functor(
        ...     ob_map={n: Dim(2)}, ar_map={}, dom=markov.Diagram, dtype=int)
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
        source = list(range(len(self.dom @ self.cod)))
        target = [
            len(self.dom) - i - 1 for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conj(np.moveaxis(self.array, source, target))
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
        >>> from sympy import Expr  # doctest: +EXTRA
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

    Calling it on a diagram converts it to a :class:`CMap` and contracts
    the network in a single ``einsum`` call under the active
    :func:`backend`, passing any optional einsum parameters through.

    Parameters:
        ob_map : The object mapping.
        ar_map : The arrow mapping.
        dom : The domain of the functor, i.e. the class of diagrams
            it evaluates, the class attribute ``dom`` by default.
        dtype : The datatype for the codomain ``Tensor[dtype]``.
        optimize : The contraction path, passed verbatim to the backend
            ``einsum``, e.g. ``"greedy"``, ``"optimal"`` or an explicit
            path.
        contract : The contraction engine, either ``"einsum"``,
            ``"opt_einsum"`` or ``"quimb"``, see :meth:`Functor.contract`.
            By default, ``einsum`` switching to ``opt_einsum`` for networks
            with more than ``config.MAX_EINSUM_INDICES`` indices.
        params : Any other optional parameter of the backend ``einsum``
            method, passed verbatim.

    Example
    -------
    >>> n, s = map(rigid.Ty, "ns")
    >>> Alice = rigid.Box('Alice', rigid.Ty(), n)
    >>> loves = rigid.Box('loves', rigid.Ty(), n.r @ s @ n.l)
    >>> Bob = rigid.Box('Bob', rigid.Ty(), n)
    >>> diagram = Alice @ loves @ Bob\\
    ...     >> rigid.Cup(n, n.r) @ s @ rigid.Cup(n.l, n)

    >>> F = Functor(
    ...     ob_map={s: 1, n: 2},
    ...     ar_map={Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]},
    ...     dom=rigid.Diagram, dtype=bool)
    >>> F(diagram)
    Tensor[bool]([True], dom=Dim(1), cod=Dim(1))

    >>> rewrite = diagram\\
    ...     .transpose_box(2).transpose_box(0, left=True).normal_form()
    >>> Equation(diagram, rewrite).draw(
    ...     figsize=(8, 3), doctest='docs/_static/tensor/rewrite.svg')

    .. image:: /_static/tensor/rewrite.svg
        :align: center

    >>> assert F(diagram) == F(rewrite)
    """
    dom, cod = frobenius.Diagram, Tensor

    def __init__(
            self, ob_map: dict[cat.Ob, Dim], ar_map: dict[cat.Box, list],
            dom: type = None, dtype: type = float,
            optimize="greedy", contract: str = None, **params):
        self.dtype, self.optimize, self.params = dtype, optimize, params
        self.contraction = contract
        cod = type(self).cod[dtype]
        super().__init__(ob_map, ar_map, dom=dom or type(self).dom, cod=cod)

    def __repr__(self):
        optimize = "" if self.optimize == "greedy"\
            else f", optimize={self.optimize!r}"
        contract = "" if self.contraction is None\
            else f", contract={self.contraction!r}"
        params = "".join(
            f", {key}={value!r}" for key, value in self.params.items())
        return factory_name(type(self))\
            + f"(ob_map={self.ob_map}, ar_map={self.ar_map}, "\
            + f"dom={factory_name(self.dom)}, "\
            + f"dtype={self.dtype.__name__}{optimize}{contract}{params})"

    def __call__(self, other):
        if isinstance(other, Dim):
            return other
        if isinstance(other, Bubble):
            return self(other.arg).map(other.func)
        if isinstance(other, (
                cat.Ob, cat.Box, monoidal.Colour, monoidal.Ty)):
            return super().__call__(other)
        if isinstance(other, cmap.CMap):
            return self.contract(other)
        assert_isinstance(other, monoidal.Diagram)
        return self.contract(cmap.CMap.from_diagram(other))

    def operands(self, other: "cmap.CMap") -> tuple[list, list, list]:
        """
        The Einstein notation for the image of a combinatorial map: a
        list of arrays, their lists of integer indices and the output
        indices, interleavable into an ``einsum`` call.

        The 2-cycles of the ``edges`` involution are the summed indices,
        boxes are the tensors and the boundary ports are the free
        indices, each carried by an identity so that every index appears
        on an array. A wire is one index of the size of its object's
        image, a loop is an identity with a repeated index, i.e. a trace.

        Parameters:
            other : The combinatorial map to translate.

        Example
        -------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> F = Functor(ob_map=lambda dim: dim,
        ...             ar_map=lambda box: box.array, dtype=int)
        >>> arrays, indices, output = F.operands(
        ...     (vector >> vector[::-1]).to_map())
        >>> for array, index in zip(arrays, indices):
        ...     print(f"{array.tolist()}, {index}")
        [0, 1], [0]
        [0, 1], [0]
        >>> assert output == []
        """
        dim = lambda typ: product(self(typ).inside)
        wires, fresh = {}, count()
        for source, target in enumerate(other.edges):
            if source <= target:
                wires[source] = wires[target] = next(fresh)
        ports, arrays, indices, output = other.ports, [], [], []
        with backend() as np:
            eye = lambda typ: np.array(
                np.eye(dim(typ)), dtype=self.dtype)
            for port in range(len(other.dom)):
                label = next(fresh)
                arrays.append(eye(ports[port].obj))
                indices.append([label, wires[port]])
                output.append(label)
            start = len(other.dom)
            for box in other.boxes:
                arity, coarity = len(box.dom), len(box.cod)
                box_ports = list(range(start, start + arity)) + list(
                    reversed(range(
                        start + arity, start + arity + coarity)))
                arrays.append(self(box).array.reshape(
                    [dim(t) for t in list(box.dom) + list(box.cod)]))
                indices.append([wires[port] for port in box_ports])
                start += arity + coarity
            for port in range(
                    other.n_ports - len(other.cod), other.n_ports):
                label = next(fresh)
                arrays.append(eye(ports[port].obj))
                indices.append([wires[port], label])
                output.append(label)
            for loop in other.loops:
                arrays.append(eye(loop))
                indices.append(2 * [next(fresh)])
        return arrays, indices, output

    def to_quimb(self, other) -> "quimb.tensor.TensorNetwork":
        """
        Translate the image of a diagram or combinatorial map to a quimb
        tensor network: one tensor per operand of :meth:`operands`, with
        the free indices ``inp0, ..., out0, ...`` named after the
        boundary ports and carried by identities in boundary order, the
        order in which quimb contraction outputs them. An index repeated
        on one array, i.e. a loop or a box traced with itself, is
        summed over beforehand.

        Parameters:
            other : The diagram or combinatorial map to translate.
        """
        import quimb.tensor as qtn
        if not isinstance(other, cmap.CMap):
            other = cmap.CMap.from_diagram(other)
        arrays, indices, output = self.operands(other)
        n_dom = len(other.dom)
        names = {label: f"inp{i}" if i < n_dom else f"out{i - n_dom}"
                 for i, label in enumerate(output)}
        tensors = []
        with backend() as np:
            for array, inds in zip(arrays, indices):
                kept = [j for j in inds if inds.count(j) == 1]
                if kept != inds:
                    array = np.einsum(array, inds, kept)
                tensors.append(qtn.Tensor(
                    array, inds=tuple(names.get(j, f"w{j}") for j in kept)))
        return qtn.TensorNetwork(tensors)

    def contract(self, other: "cmap.CMap") -> Tensor:
        """
        Contract the image of a combinatorial map, read as Einstein
        notation by :meth:`operands`, under the active :func:`backend`.

        The engine is chosen by the ``contract`` parameter of the
        functor: ``"einsum"`` calls the backend ``einsum``,
        ``"opt_einsum"`` the optional package of the same name and
        ``"quimb"`` contracts the network of :meth:`to_quimb`, where
        ``optimize`` may be a ``cotengra`` path optimizer and a
        ``max_bond`` parameter or a compressed optimizer selects
        approximate contraction. By default, ``einsum`` is used and
        networks with more than ``config.MAX_EINSUM_INDICES`` indices
        switch to ``opt_einsum``.

        Parameters:
            other : The combinatorial map to contract.
        """
        result = lambda array: self.cod(
            array, self(other.dom), self(other.cod))
        n_indices = other.n_edges + len(other.dom) + len(other.cod)
        contraction = self.contraction or (
            "einsum" if n_indices <= config.MAX_EINSUM_INDICES
            else "opt_einsum")
        if contraction == "quimb":
            return result(self.contract_quimb(other))
        if contraction not in ("einsum", "opt_einsum"):
            raise ValueError(
                f"Expected 'einsum', 'opt_einsum' or 'quimb', "
                f"got {contraction!r}.")
        arrays, indices, output = self.operands(other)
        if not arrays:
            return result([1])
        operands = [x for pair in zip(arrays, indices) for x in pair]
        if contraction == "opt_einsum":
            import opt_einsum
            return result(opt_einsum.contract(
                *operands, output, optimize=self.optimize, **self.params))
        with backend() as np:
            params = dict(self.params, optimize=self.optimize)\
                if isinstance(get_backend(), (NumPy, JAX))\
                else self.params
            return result(np.einsum(*operands, output, **params))

    def contract_quimb(self, other: "cmap.CMap"):
        """
        Contract the network of :meth:`to_quimb` and return its array.

        Gradients of jax and pytorch arrays survive the contraction
        with ``autoray >= 0.9``, whose older versions fall back to
        numpy when reusing a cached contraction tree.

        Parameters:
            other : The combinatorial map to contract.
        """
        import quimb.tensor as qtn
        network = self.to_quimb(other)
        if not network.tensors:
            return [1]
        compressed = "max_bond" in self.params\
            or getattr(self.optimize, "compressed", False)\
            or "Compressed" in type(self.optimize).__name__
        for tensor in network.tensors if compressed else ():
            if getattr(tensor.data.dtype, "kind", "") in "?bui":
                tensor.modify(data=tensor.data.astype("complex128"))
        output_inds = [f"inp{i}" for i in range(len(other.dom))]\
            + [f"out{i}" for i in range(len(other.cod))]
        method = network.contract_compressed if compressed\
            else network.contract
        result = method(
            output_inds=output_inds, optimize=self.optimize, **self.params)
        return result.data if isinstance(result, qtn.Tensor) else result


@factory
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

    def eval(self, dtype: type = None, optimize="greedy",
             contract: str = None, **params) -> Tensor:
        """
        Evaluate a tensor network as a :class:`Tensor`: call the
        :class:`Functor` that sends each box to its array.

        Parameters:
            dtype : The datatype for spiders and the result,
                inferred from the boxes by default.
            optimize : The contraction path, passed verbatim to the
                engine.
            contract : The contraction engine, either ``"einsum"``,
                ``"opt_einsum"`` or ``"quimb"``, see
                :meth:`Functor.contract`. By default, ``einsum``
                switching to ``opt_einsum`` for networks with more than
                ``config.MAX_EINSUM_INDICES`` indices.
            params : Any other optional parameter of the engine,
                passed verbatim.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).eval().array == 1
        >>> assert (vector >> vector[::-1]).eval(
        ...     optimize="optimal").array == 1
        >>> assert (vector >> vector[::-1]).eval(  # doctest: +EXTRA
        ...     contract="quimb").array == 1
        """
        return Functor(
            ob_map=lambda x: Dim(*(
                getattr(obj, "dim", obj) for obj in x.inside)),
            ar_map=lambda box: box.array,
            dtype=dtype or getattr(self, "dtype", None), optimize=optimize,
            contract=contract, **params)(self)

    def to_quimb(self, dtype: type = None) -> "quimb.tensor.TensorNetwork":
        """
        Convert a tensor diagram to a quimb tensor network: call the
        :meth:`Functor.to_quimb` of the functor that sends each box to
        its array.

        The boundary ports are the free indices, named ``inp0, ...``
        and ``out0, ...`` in boundary order.

        Parameters:
            dtype : The datatype for spiders and boundary identities,
                inferred from the boxes by default.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> t_net = (vector >> vector[::-1]).to_quimb()  # doctest: +EXTRA
        >>> assert t_net.contract(preserve_tensor=True).data == 1
        """
        return Functor(
            ob_map=lambda x: Dim(*(
                getattr(obj, "dim", obj) for obj in x.inside)),
            ar_map=lambda box: box.array,
            dtype=dtype or self.dtype).to_quimb(self)

    def to_tn(self, dtype: type = None) -> tuple[
            list["tensornetwork.Node"], list["tensornetwork.Edge"]]:
        """
        Convert a tensor diagram to :code:`tensornetwork`.

        Parameters:
            dtype : Used for spiders.

        Examples
        --------
        >>> import numpy as np
        >>> from tensornetwork import Node, Edge  # doctest: +EXTRA
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
            if isinstance(box, Permutation):
                segment = outputs[offset:offset + len(box.dom)]
                outputs[offset:offset + len(box.dom)] = [
                    segment[i] for i in box.perm]
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
        left, box, right = self.inside[0].boxes_and_types
        tail = self[1:]
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
        >>> from sympy import Expr  # doctest: +EXTRA
        >>> from sympy.abc import x, y, z
        >>> vector = Box("v", Dim(1), Dim(2), [x ** 2, y * z])
        >>> vector.jacobian([x, y, z]).eval(dtype=Expr)
        Tensor[Expr]([2.0*x, 0, 0, 1.0*z, 0, 1.0*y], dom=Dim(1), cod=Dim(3, 2))
        """
        dim = Dim(len(variables) or 1)
        result = Sum((), self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = Tensor.zero(Dim(1), dim)
            onehot.array[i] = 1
            result += Box(str(var), Dim(1), dim, onehot.array) @ self.grad(var)
        return result


CMap = cmap.CMap[Diagram]
# NamedGeneric caches CMap[Diagram] by type parameter (abc.NamedGeneric),
# and Diagram.to_map looks it up the same way, so a subclass here would be
# invisible to to_map: attributes must land on the cached class itself.
CMap.dtype = None
CMap.to_quimb = Diagram.to_quimb


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
            cls[dtype], name, dom, cod, data, *args, **kwargs)

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

    def setoid(self):
        """ Compare boxes by turning their internal `data` into tuples. """
        data = () if self.data is None else\
            tuple(self.data) if isinstance(self.data, list) else (self.data, )
        return (self.name, self.dom, self.cod, self.dtype) + data


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


class Permutation(frobenius.Permutation, Box):
    "A permutation in a tensor diagram."

    @property
    def array(self):
        doms = [Dim(getattr(dim.inside[0], 'dim', dim.inside[0]))
                for dim in self.dom]
        return Tensor.permutation(self.perm, doms).array


class Swap(Permutation, frobenius.Swap, Box):
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
    >>> Equation(vector >> spider, vector @ vector).draw(figsize=(3, 2),
    ...     doctest='docs/_static/tensor/frobenius-example.svg')

    .. image:: /_static/tensor/frobenius-example.svg
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
    ...                     doctest='docs/_static/tensor/men-are-mortal.svg')

    .. image:: /_static/tensor/men-are-mortal.svg
        :align: center

    >>> from sympy import Expr  # doctest: +EXTRA
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

    >>> Equation(lhs, rhs).draw(figsize=(5, 2), wire_labels=False,
    ...                         doctest='docs/_static/tensor/product-rule.svg')

    .. image:: /_static/tensor/product-rule.svg
        :align: center
    """

    def __init__(self, inside, func=lambda x: int(not x), **params):
        self.func = func
        super().__init__(inside, **params)

    def grad(self, var, **params):
        """
        The gradient of a bubble is given by the chain rule.

        >>> from sympy.abc import x  # doctest: +EXTRA
        >>> g = Box('g', Dim(2), Dim(2), [2 * x, 0, 0, x + 1])
        >>> f = lambda d: d.bubble(func=lambda x: x ** 2, drawing_name="f")
        >>> lhs, rhs = Box.grad(f(g), x), f(g).grad(x)

        >>> Equation(lhs, rhs).draw(wire_labels=False,
        ...     doctest='docs/_static/tensor/chain-rule.svg')

        .. image:: /_static/tensor/chain-rule.svg
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


Diagram.sum_factory, Diagram.swap_factory = Sum, Swap
Diagram.permutation_factory = Permutation
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.spider_factory, Diagram.bubble_factory = Spider, Bubble
Id = Diagram.id


class Equation(frobenius.Equation):
    """ The :class:`frobenius.Equation` of tensor diagrams. """
