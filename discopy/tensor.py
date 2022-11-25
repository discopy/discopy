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
    Box
    Swap
    Cup
    Cap
    Spider
    Sum
    Bubble

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        backend
        get_backend

Example
-------
>>> n = Ty('n')
>>> Alice, Bob = Box('Alice', Ty(), n), Box('Bob', Ty(), n)
>>> loves = Box('loves', n, n)
>>> ob, ar = {n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = Functor(ob, ar)
>>> assert F(Alice >> loves >> Bob.dagger()) == 1
"""

from __future__ import annotations
from contextlib import contextmanager

from discopy import (
    cat, config, messages, monoidal, rigid, symmetric, frobenius)
from discopy.cat import Composable, AxiomError, factory
from discopy.monoidal import Whiskerable
from discopy.frobenius import Ob, Ty, Cup, Cap, Category
from discopy.utils import assert_isinstance, assert_isatomic, product


@factory
class Dim(Ty):
    """
    A dimension is a tuple of positive integers
    with product ``@`` and unit ``Dim(1)``.

    Example
    -------
    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    ob_factory = int

    def __init__(self, *inside: int):
        for dim in inside:
            assert_isinstance(dim, int)
            if dim < 1:
                raise ValueError
        super().__init__(*(dim for dim in inside if dim > 1))

    def __repr__(self):
        return "Dim({})".format(', '.join(map(repr, self.inside)) or '1')

    __str__ = __repr__
    l = r = property(lambda self: self.factory(*self.inside[::-1]))


class Tensor(Composable, Whiskerable):
    """
    A tensor is an ``array`` and a pair of dimensions ``dom`` and ``cod``.

    Parameters:
        inside : The array inside the tensor.
        dom : The domain dimension.
        cod : The codomain dimension.

    Examples
    --------
    >>> m = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
    >>> v = Tensor([0, 1], Dim(1), Dim(2))
    >>> v >> m >> v.dagger()
    Tensor([0], dom=Dim(1), cod=Dim(1))

    Notes
    -----
    Tensors can have sympy symbols as free variables.

    >>> from sympy.abc import phi, psi
    >>> v = Tensor([phi, psi], Dim(1), Dim(2))
    >>> d = v >> v.dagger()
    >>> assert v >> v.dagger() == Tensor(
    ...     [phi * phi.conjugate() + psi * psi.conjugate()], Dim(1), Dim(1))

    These can be substituted and lambdifed.

    >>> v.subs(phi, 0).lambdify(psi)(1)
    Tensor([0, 1], dom=Dim(1), cod=Dim(2))

    We can also use jax.numpy using Tensor.backend.

    >>> with backend('jax'):
    ...     f = lambda *xs: d.lambdify(phi, psi)(*xs).array
    ...     import jax
    ...     assert jax.grad(f)(1., 2.) == 2.
    """
    def __init__(self, array: "array", dom: Dim, cod: Dim):
        assert_isinstance(dom, Dim)
        assert_isinstance(cod, Dim)
        self.dom, self.cod = dom, cod
        with backend() as np:
            self.array = np.array(array).reshape(dom.inside + cod.inside)

    def __iter__(self):
        for i in self.array:
            yield i

    def __bool__(self):
        return bool(self.array)

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    def __repr__(self):
        np_array = getattr(self.array, 'numpy', lambda: self.array)()
        return "Tensor({}, dom={}, cod={})".format(
            array2string(np_array.reshape(-1)), self.dom, self.cod)

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        if other == 0:
            return self
        assert_isinstance(other, Tensor)
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Tensor(self.array + other.array, self.dom, self.cod)

    __radd__ = __add__

    def __eq__(self, other):
        with backend() as np:
            if not isinstance(other, Tensor):
                return np.all(np.array(self.array == other))
            return (self.dom, self.cod) == (other.dom, other.cod)\
                and np.all(np.array(self.array == other.array))

    @staticmethod
    def id(dom=Dim(1)) -> Tensor:
        with backend() as np:
            return Tensor(np.eye(product(dom.inside)), dom, dom)

    def then(self, other: Tensor = None, *others: Tensor) -> Tensor:
        if other is None or others:
            return Diagram.then(self, other, *others)
        assert_isinstance(other, Tensor)
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        with backend() as np:
            array = np.tensordot(self.array, other.array, len(self.cod))\
                if self.array.shape and other.array.shape\
                else self.array * other.array
        return Tensor(array, self.dom, other.cod)

    def tensor(self, other: Tensor = None, *others: Tensor) -> Tensor:
        if other is None or others:
            return Diagram.tensor(self, other, *others)
        assert_isinstance(other, Tensor)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        source = range(len(dom @ cod))
        target = [
            i if i < len(self.dom) or i >= len(self.dom @ self.cod @ other.dom)
            else i - len(self.cod) if i >= len(self.dom @ self.cod)
            else i + len(other.dom) for i in source]
        with backend() as np:
            array = np.tensordot(self.array, other.array, 0)\
                if self.array.shape and other.array.shape\
                else self.array * other.array
            array = np.moveaxis(array, source, target)
        return Tensor(array, dom, cod)

    def dagger(self) -> Tensor:
        source = range(len(self.dom @ self.cod))
        target = [i + len(self.cod) if i < len(self.dom) else
                  i - len(self.dom) for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conjugate(np.moveaxis(self.array, source, target))
        return Tensor(array, self.cod, self.dom)

    @staticmethod
    def cups(left: Dim, right: Dim) -> Tensor:
        assert_isinstance(left, Dim)
        assert_isinstance(right, Dim)
        if left.r != right:
            raise AxiomError
        return rigid.nesting(Tensor, lambda x, y:
            Tensor(Tensor.id(left).array, x @ y, Dim(1)))(left, right)

    @staticmethod
    def caps(left: Dim, right: Dim) -> Tensor:
        return Tensor.cups(left, right).dagger()

    @staticmethod
    def swap(left: Dim, right: Dim) -> Tensor:
        dom, cod = left @ right, right @ left
        array = Tensor.id(dom).array
        source = range(len(dom), 2 * len(dom))
        target = [i + len(right) if i < len(dom @ left)
                  else i - len(left) for i in source]
        with backend() as np:
            return Tensor(np.moveaxis(array, source, target), dom, cod)

    @staticmethod
    def spider_factory(
            n_legs_in: int, n_legs_out: int, typ: Dim, phase=None) -> Tensor:
        if phase is not None:
            raise NotImplementedError
        assert_isatomic(typ, Dim)
        n, = typ.inside
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        result = Tensor.zeros(dom, cod)
        for i in range(n):
            result.array[len(dom @ cod) * (i, )] = 1
        return result

    @staticmethod
    def spiders(
            n_legs_in: int, n_legs_out: int, typ: Dim, phase=None) -> Tensor:
        """
        The tensor of interleaving spiders.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return frobenius.Diagram.spiders.__func__(
            Tensor, n_legs_in, n_legs_out, typ, phase)

    def transpose(self, left=False) -> Tensor:
        """
        Returns the diagrammatic transpose.

        Note
        ----
        This is *not* the same as the algebraic transpose for complex dims.
        """
        return Tensor(self.array.transpose(), self.cod[::-1], self.dom[::-1])

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
                return Tensor(np.conjugate(self.array), self.dom, self.cod)
        # reverse the wires for both inputs and outputs
        source = range(len(self.dom @ self.cod))
        target = [
            len(self.dom) - i - 1 for i in range(len(self.dom @ self.cod))]
        with backend() as np:
            array = np.conjugate(np.moveaxis(self.array, source, target))
        return Tensor(array, self.dom[::-1], self.cod[::-1])

    l = r = property(conjugate)

    def round(self, decimals=0) -> Tensor:
        """ Rounds the entries of a tensor up to a number of decimals. """
        with backend() as np:
            array = np.around(self.array, decimals=decimals)
        return Tensor(array, self.dom, self.cod)

    def map(self, func: Callable) -> Tensor:
        """ Apply a function elementwise. """
        array = list(map(func, self.array.reshape(-1)))
        return Tensor(array, self.dom, self.cod)

    @staticmethod
    def zeros(dom: Dim, cod: Dim) -> Tensor:
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zeros(Dim(2), Dim(2))\\
        ...     == Tensor([0, 0, 0, 0], Dim(2), Dim(2))
        """
        with backend() as np:
            return Tensor(np.zeros((dom @ cod).inside), dom, cod)

    def subs(self, *args) -> Tensor:
        return self.map(lambda x: getattr(x, "subs", lambda y, *_: y)(*args))

    def grad(self, var, **params) -> Tensor:
        """ Gradient with respect to variables. """
        return self.map(lambda x:
                        getattr(x, "diff", lambda _: 0)(var, **params))

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
        >>> from sympy.abc import x, y, z
        >>> vector = Tensor([x ** 2, y * z], Dim(1), Dim(2))
        >>> vector.jacobian(x, y, z)
        Tensor([2.0*x, 0, 0, 1.0*z, 0, 1.0*y], dom=Dim(1), cod=Dim(3, 2))
        """
        dim = Dim(len(variables) or 1)
        result = Tensor.zeros(self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = Tensor.zeros(Dim(1), dim)
            onehot.array[i] = 1
            result += onehot @ self.grad(var)
        return result

    def lambdify(self, *symbols: "sympy.Symbol", **kwargs) -> Callable:
        from sympy import lambdify
        with backend() as np:
            array = lambdify(symbols, self.array, modules=np.module, **kwargs)
        return lambda *xs: Tensor(array(*xs), self.dom, self.cod)


class Functor(frobenius.Functor):
    """ Implements a tensor-valued frobenius functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = Box('f', x, x @ y)
    >>> F = Functor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Tensor([0, 1], dom=Dim(1), cod=Dim(2))
    """
    cod = Category(Dim, Tensor)

    def __call__(self, other):
        if isinstance(other, (cat.Sum)):
            return super().__call__(other)
        if isinstance(other, Bubble):
            return self(other.arg).map(other.func)
        if isinstance(other, monoidal.Ty):
            def obj_to_dim(obj):
                if isinstance(obj, rigid.Ob) and obj.z != 0:
                    obj = type(obj)(obj.name)  # sets z=0
                result = self.ob[type(other)(obj)]
                if isinstance(result, int):
                    result = Dim(result)
                if not isinstance(result, Dim):
                    result = Dim.upgrade(result)
                return result
            return Dim(1).tensor(*map(obj_to_dim, other.inside))
        if isinstance(other, monoidal.Box)\
                and not isinstance(other, symmetric.Swap):
            if other.z % 2 != 0:
                while other.z != 0:
                    other = other.l if other.z > 0 else other.r
                return self(other).conjugate()
            if other.is_dagger:
                return self(other.dagger()).dagger()
            return Tensor(self.ar[other], self(other.dom), self(other.cod))
        assert_isinstance(other, monoidal.Diagram)

        def dim(scan):
            return len(self(scan))
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
        return Tensor(array, self(other.dom), self(other.cod))


@factory
class Diagram(frobenius.Diagram):
    """
    A tensor diagram is a frobenius diagram with tensor boxes.

    Examples
    --------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> diagram = vector[::-1] >> vector @ vector
    >>> print(diagram)
    vector[::-1] >> vector >> Dim(2) @ vector
    """
    def eval(self, contractor: Callable = None, dtype: type = None) -> Tensor:
        """
        Evaluate a tensor diagram as a :class:`Tensor`.

        Parameters:
            contractor : Use ``tensornetwork`` or :class:`Functor` by default.
            dtype : Used for spiders.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).eval() == 1
        >>> import tensornetwork as tn
        >>> assert (vector >> vector[::-1]).eval(tn.contractors.auto) == 1
        """
        if contractor is None and "numpy" not in get_backend().__package__:
            raise Exception(
                'Provide a tensornetwork contractor'
                'when using a non-numpy backend.')
        if contractor is None:
            return Functor(ob=lambda x: x, ar=lambda f: f.array)(self)
        array = contractor(*self.to_tn(dtype=dtype)).tensor
        return Tensor(array, self.dom, self.cod)

    def to_tn(self, dtype: type = None
            ) -> tuple[list["tensornetwork.Node"], list["tensornetwork.Edge"]]:
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
            dtype = self._infer_dtype()
        nodes = [
            tn.CopyNode(2, getattr(dim, 'dim', dim), f'input_{i}', dtype=dtype)
            for i, dim in enumerate(self.dom.inside)]
        inputs, outputs = [n[0] for n in nodes], [n[1] for n in nodes]
        for box, offset in zip(self.boxes, self.offsets):
            if isinstance(box, symmetric.Swap):
                outputs[offset], outputs[offset + 1]\
                    = outputs[offset + 1], outputs[offset]
                continue
            if isinstance(box, Spider):
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
                array = box.eval().array
                node = tn.Node(array, str(box))
            for i, _ in enumerate(box.dom):
                tn.connect(outputs[offset + i], node[i])
            outputs[offset:offset + len(box.dom)] = node[len(box.dom):]
            nodes.append(node)
        return nodes, inputs + outputs

    def _infer_dtype(self):
        for box in self.boxes:
            if not isinstance(box, (Spider, symmetric.Swap)):
                array = box.array
                while True:
                    # minimise data to potentially copy
                    try:
                        array = array[0]
                    except IndexError:
                        break
                try:
                    import numpy
                    return numpy.asarray(array).dtype
                except (RuntimeError, TypeError):
                    # assume that the array is actually a PyTorch tensor
                    return array.detach().cpu().numpy().dtype
        else:
            import numpy
            return numpy.float64

    def grad(self, var, **params):
        """ Gradient with respect to :code:`var`. """
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        left, box, right, tail = tuple(self.layers[0]) + (self[1:], )
        t1 = self.id(left) @ box.grad(var, **params) @ self.id(right) >> tail
        t2 = self.id(left) @ box @ self.id(right) >> tail.grad(var, **params)
        return t1 + t2

    def jacobian(self, variables, **params):
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
        >>> from sympy.abc import x, y, z
        >>> vector = Box("v", Dim(1), Dim(2), [x ** 2, y * z])
        >>> vector.jacobian([x, y, z]).eval()
        Tensor([2.0*x, 0, 0, 1.0*z, 0, 1.0*y], dom=Dim(1), cod=Dim(3, 2))
        """
        dim = Dim(len(variables) or 1)
        result = Sum((), self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = Tensor.zeros(Dim(1), dim)
            onehot.array[i] = 1
            result += Box(var, Dim(1), dim, onehot.array) @ self.grad(var)
        return result


class Box(frobenius.Box, Diagram):
    """
    A tensor box is a frobenius box with an array as data.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. its input dimension.
        cod : The codomain of the box, i.e. its output dimension.
        array : The array inside the tensor box.
    """
    __ambiguous_inheritance__ = (frobenius.Box, )

    def __init__(self, name: str, dom: Dim, cod: Dim, array, **params):
        if array is not None:
            with backend() as np:
                array = np.array(data).reshape(dom.inside + cod.inside)
        self.array = array
        frobenius.Box.__init__(name, dom, cod, **params)

    def grad(self, var, **params):
        return self.bubble(
            func=lambda x: getattr(x, "diff", lambda _: 0)(var),
            drawing_name="$\\partial {}$".format(var))

    def __eq__(self, other):
        if isinstance(other, Box):
            with backend() as np:
                return np.all(np.array(self.array == other.array))\
                    and (self.name, self.dom, self.cod)\
                    == (other.name, other.dom, other.cod)
        return isinstance(other, Diagram)\
            and other.inside == (self.layer_factory.cast(self), )

    def __hash__(self):
        return hash((self.name, self.dom, self.cod, str(self.array)))


class Cup(frobenius.Cup, Box):
    """
    A tensor cup is a frobenius cup in a tensor diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (frobenius.Cup, )


class Cap(frobenius.Cap, Box):
    """
    A tensor cap is a frobenius cap in a tensor diagram.

    Parameters:
        left (Ty) : The atomic type.
        right (Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (frobenius.Cap, )


class Swap(frobenius.Swap, Box):
    """
    A tensor swap is a frobenius swap in a tensor diagram.

    Parameters:
        left (Dim) : The type on the top left and bottom right.
        right (Dim) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (frobenius.Swap, )

    @property
    def array(self):
        return Tensor.swap(self.left, self.right).array


class Spider(frobenius.Spider, Box):
    """
    A tensor spider is a frobenius spider in a tensor diagram.

    Parameters:
        n_legs_in : The number of legs in.
        n_legs_out : The number of legs out.
        typ : The type of the spider.

    Examples
    --------
    >>> vector = Box('vec', Dim(1), Dim(2), [0, 1])
    >>> spider = Spider(1, 2, Dim(2))
    >>> assert (vector >> spider).eval() == (vector @ vector).eval()
    >>> from discopy import drawing
    >>> drawing.equation(vector >> spider, vector @ vector, figsize=(3, 2),\\
    ... path='docs/_static/imgs/tensor/frobenius-example.png')

    .. image:: ../_static/imgs/tensor/frobenius-example.png
        :align: center
    """
    __ambiguous_inheritance__ = (frobenius.Spider, )

    @property
    def array(self):
        return Tensor.spiders(len(self.dom), len(self.cod), self.typ).array


class Sum(monoidal.Sum, Box):
    """
    A formal sum of tensor diagrams with the same domain and codomain.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Dim) : The domain of the formal sum.
        cod (Dim) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (monoidal.Sum, )

    def eval(self, contractor=None):
        return sum(term.eval(contractor=contractor) for term in self.terms)


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
    >>> assert men_are_mortal.eval()
    >>> men_are_mortal.draw(draw_type_labels=False,
    ...                     path='docs/_static/imgs/tensor/men-are-mortal.png')

    .. image:: ../_static/imgs/tensor/men-are-mortal.png
        :align: center

    >>> from sympy.abc import x
    >>> f = Box('f', Dim(2), Dim(2), [1, 0, 0, x])
    >>> g = Box('g', Dim(2), Dim(2), [-x, 0, 0, 1])
    >>> def grad(diagram, var):
    ...     return diagram.bubble(
    ...         func=lambda x: getattr(x, "diff", lambda _: 0)(var),
    ...         drawing_name="d${}$".format(var))
    >>> lhs = grad(f >> g, x)
    >>> rhs = (grad(f, x) >> g) + (f >> grad(g, x))
    >>> assert lhs.eval() == rhs.eval()
    >>> from discopy import drawing
    >>> drawing.equation(lhs, rhs, figsize=(5, 2), draw_type_labels=False,
    ...                  path='docs/_static/imgs/tensor/product-rule.png')

    .. image:: ../_static/imgs/tensor/product-rule.png
        :align: center
    """
    __ambiguous_inheritance__ = (monoidal.Bubble, )

    def __init__(self, inside, func=lambda x: int(not x), **params):
        self.func = func
        super().__init__(inside, **params)

    def grad(self, var, **params):
        """
        The gradient of a bubble is given by the chain rule.

        >>> from sympy.abc import x
        >>> from discopy import drawing
        >>> g = Box('g', Dim(2), Dim(2), [2 * x, 0, 0, x + 1])
        >>> f = lambda d: d.bubble(func=lambda x: x ** 2, drawing_name="f")
        >>> lhs, rhs = Box.grad(f(g), x), f(g).grad(x)
        >>> drawing.equation(lhs, rhs, draw_type_labels=False,
        ...                  path='docs/_static/imgs/tensor/chain-rule.png')

        .. image:: ../_static/imgs/tensor/chain-rule.png
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


def array2string(array, **params):
    """ Numpy array pretty print. """
    import numpy
    numpy.set_printoptions(threshold=config.NUMPY_THRESHOLD)
    return numpy.array2string(array, **dict(params, separator=', '))\
        .replace('[ ', '[').replace('  ', ' ')


class Backend:
    def __init__(self, module, array=None):
        self.module, self.array = module, array or module.array

    def __getattr__(self, attr):
        return getattr(self.module, attr)


class NumPy(Backend):
    def __init__(self):
        import numpy
        super().__init__(numpy)


class JAX(Backend):
    def __init__(self):
        import jax
        super().__init__(jax.numpy)


class PyTorch(Backend):
    def __init__(self):
        import torch
        super().__init__(torch, array=torch.as_tensor)


class TensorFlow(Backend):
    def __init__(self):
        import tensorflow.experimental.numpy as tnp
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
        super().__init__(tnp)


BACKENDS = {'np': NumPy,
            'numpy': NumPy,
            'jax': JAX,
            'jax.numpy': JAX,
            'pytorch': PyTorch,
            'torch': PyTorch,
            'tensorflow': TensorFlow,
}

@contextmanager
def backend(name=None, _stack=[config.DEFAULT_BACKEND], _cache=dict()):
    name = name or _stack[-1]
    _stack.append(name)
    try:
        if name not in _cache:
            _cache[name] = BACKENDS[name]()
        yield _cache[name]
    finally:
        _stack.pop()

def get_backend():
    with backend() as result:
        return result

for cls in [Diagram, Box, Swap, Spider, Sum, Bubble]:
    cls.ty_factory = Dim

Diagram.braid_factory = Swap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.spider_factory, Diagram.bubble_factory = Spider, Bubble
Diagram.sum_factory = Sum
Id = Diagram.id
