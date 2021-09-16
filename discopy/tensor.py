# -*- coding: utf-8 -*-

"""
Implements dagger monoidal functors into tensors.

>>> n = Ty('n')
>>> Alice, Bob = rigid.Box('Alice', Ty(), n), rigid.Box('Bob', Ty(), n)
>>> loves = rigid.Box('loves', n, n)
>>> ob, ar = {n: 2}, {Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]}
>>> F = Functor(ob, ar)
>>> assert F(Alice >> loves >> Bob.dagger()) == 1
"""

import numpy

from discopy import cat, config, messages, monoidal, rigid
from discopy.cat import AxiomError
from discopy.rigid import Ob, Ty, Cup, Cap


if config.IMPORT_JAX:  # pragma: no cover
    import warnings
    for msg in config.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore", message=msg)
    import jax

numpy.set_printoptions(threshold=config.NUMPY_THRESHOLD)


def array2string(array, **params):
    """ Numpy array pretty print. """
    return numpy.array2string(array, **dict(params, separator=', '))\
        .replace('[ ', '[').replace('  ', ' ')


class Dim(Ty):
    """ Implements dimensions as tuples of positive integers.
    Dimensions form a monoid with product @ and unit Dim(1).

    >>> Dim(1) @ Dim(2) @ Dim(3)
    Dim(2, 3)
    """
    @staticmethod
    def upgrade(old):
        return Dim(*[x.name for x in old.objects])

    def __init__(self, *dims):
        dims = map(lambda x: x if isinstance(x, monoidal.Ob) else Ob(x), dims)
        dims = list(filter(lambda x: x.name != 1, dims))  # Dim(1) == Dim()
        for dim in dims:
            if not isinstance(dim.name, int):
                raise TypeError(messages.type_err(int, dim.name))
            if dim.name < 1:
                raise ValueError
        super().__init__(*dims)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().__getitem__(key)
        return super().__getitem__(key).name

    def __repr__(self):
        return "Dim({})".format(', '.join(map(repr, self)) or '1')

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))

    @property
    def l(self):
        """
        >>> assert Dim(2, 3, 4).l == Dim(4, 3, 2)
        """
        return Dim(*self[::-1])

    @property
    def r(self):
        """
        >>> assert Dim(2, 3, 4).r == Dim(4, 3, 2)
        """
        return Dim(*self[::-1])


class Tensor(rigid.Box):
    """ Implements a tensor with dom, cod and numpy array.

    Examples
    --------
    >>> m = Tensor(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Tensor(Dim(1), Dim(2), [0, 1])
    >>> v >> m >> v.dagger()
    Tensor(dom=Dim(1), cod=Dim(1), array=[0])

    Notes
    -----
    Tensors can have sympy symbols as free variables.

    >>> from sympy.abc import phi, psi
    >>> v = Tensor(Dim(1), Dim(2), [phi, psi])
    >>> d = v >> v.dagger()
    >>> assert v >> v.dagger() == Tensor(
    ...     Dim(1), Dim(1), [phi * phi.conjugate() + psi * psi.conjugate()])

    These can be substituted and lambdifed.

    >>> v.subs(phi, 0).lambdify(psi)(1)
    Tensor(dom=Dim(1), cod=Dim(2), array=[0, 1])

    We can also use jax.numpy by changing the class variable :code:`Tensor.np`.

    >>> from contextlib import contextmanager
    >>> import jax
    >>> @contextmanager
    ... def jaxify():
    ...     Tensor.np, tmp = jax.numpy, Tensor.np
    ...     yield
    ...     Tensor.np = tmp
    >>> with jaxify():
    ...     f = lambda *xs: d.lambdify(phi, psi)(*xs).array[0]
    ...     assert jax.grad(f)(1., 2.) == 2.
    """
    np = jax.numpy if config.IMPORT_JAX else numpy

    def __init__(self, dom, cod, array):
        self._array = Tensor.np.array(array).reshape(tuple(dom @ cod) or (1, ))
        super().__init__("Tensor", dom, cod)

    def __iter__(self):
        for i in self.array:
            yield i

    @property
    def array(self):
        """ Numpy array. """
        return self._array

    def __bool__(self):
        return bool(self.array)

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    def __repr__(self):
        return "Tensor(dom={}, cod={}, array={})".format(
            self.dom, self.cod, array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        return Tensor(self.dom, self.cod, self.array + other.array)

    def __radd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return Tensor.np.all(self.array == other)
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and Tensor.np.all(self.array == other.array)

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other, = others
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = Tensor.np.tensordot(self.array, other.array, len(self.cod))\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        return Tensor(self.dom, other.cod, array)

    def tensor(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        array = Tensor.np.tensordot(self.array, other.array, 0)\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        source = range(len(dom @ cod))
        target = [
            i if i < len(self.dom) or i >= len(self.dom @ self.cod @ other.dom)
            else i - len(self.cod) if i >= len(self.dom @ self.cod)
            else i + len(other.dom) for i in source]
        return Tensor(dom, cod, Tensor.np.moveaxis(array, source, target))

    def dagger(self):
        array = Tensor.np.moveaxis(
            self.array, range(len(self.dom @ self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom @ self.cod))])
        return Tensor(self.cod, self.dom, Tensor.np.conjugate(array))

    @staticmethod
    def id(dom=Dim(1)):
        from numpy import prod
        return Tensor(dom, dom, Tensor.np.identity(int(prod(dom))))

    @staticmethod
    def cups(left, right):
        return rigid.cups(
            left, right, ar_factory=Tensor,
            cup_factory=lambda left, right:
                Tensor(left @ right, Dim(1), Tensor.id(left).array))

    @staticmethod
    def caps(left, right):
        return Tensor.cups(left, right).dagger()

    @staticmethod
    def swap(left, right):
        array = Tensor.id(left @ right).array
        source = range(len(left @ right), 2 * len(left @ right))
        target = [i + len(right) if i < len(left @ right @ left)
                  else i - len(left) for i in source]
        return Tensor(left @ right, right @ left,
                      Tensor.np.moveaxis(array, source, target))

    def transpose(self, left=False):
        """
        Returns the diagrammatic transpose.

        Note
        ----
        This is *not* the same as the algebraic transpose for complex dims.
        """
        return Tensor(self.cod[::-1], self.dom[::-1], self.array.transpose())

    def conjugate(self):
        """
        Returns the diagrammatic conjugate of a tensor.

        Note
        ----
        This is *not* the same as the algebraic conjugate for complex dims.
        """
        # reverse the wires for both inputs and outputs
        dom, cod = self.dom, self.cod
        array = Tensor.np.moveaxis(
            self.array, range(len(dom @ cod)),
            [len(dom) - i - 1 for i in range(len(dom @ cod))])
        return Tensor(dom[::-1], cod[::-1], Tensor.np.conjugate(array))

    l = r = property(conjugate)

    def round(self, decimals=0):
        """ Rounds the entries of a tensor up to a number of decimals. """
        return Tensor(self.dom, self.cod,
                      Tensor.np.around(self.array, decimals=decimals))

    def map(self, func):
        """ Apply a function elementwise. """
        return Tensor(
            self.dom, self.cod, list(map(func, self.array.flatten())))

    @staticmethod
    def zeros(dom, cod):
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zeros(Dim(2), Dim(2))\\
        ...     == Tensor(Dim(2), Dim(2), [0, 0, 0, 0])
        """
        return Tensor(dom, cod, Tensor.np.zeros(dom @ cod))

    def subs(self, *args):
        return self.map(lambda x: getattr(x, "subs", lambda y, *_: y)(*args))

    def grad(self, var, **params):
        """ Gradient with respect to variables. """
        return self.map(lambda x:
                        getattr(x, "diff", lambda _: 0)(var, **params))

    def jacobian(self, variables, **params):
        """
        Jacobian with respect to :code:`variables`.

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
        >>> vector = Tensor(Dim(1), Dim(2), [x ** 2, y * z])
        >>> vector.jacobian([x, y, z])
        Tensor(dom=Dim(1), cod=Dim(3, 2), array=[2.0*x, 0, 0, 1.0*z, 0, 1.0*y])
        """
        dim = Dim(len(variables) or 1)
        result = Tensor.zeros(self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = numpy.zeros(dim or (1, ))
            onehot[i] = 1
            result += Tensor(Dim(1), dim, onehot) @ self.grad(var)
        return result

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify
        array = lambdify(
            symbols, self.array, **dict({'modules': Tensor.np}, **kwargs))
        return lambda *xs: Tensor(self.dom, self.cod, array(*xs))


class Functor(rigid.Functor):
    """ Implements a tensor-valued rigid functor.

    >>> x, y = Ty('x'), Ty('y')
    >>> f = rigid.Box('f', x, x @ y)
    >>> F = Functor({x: 1, y: 2}, {f: [0, 1]})
    >>> F(f)
    Tensor(dom=Dim(1), cod=Dim(2), array=[0, 1])
    """
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=Dim, ar_factory=Tensor)

    def __repr__(self):
        return super().__repr__().replace("Functor", "tensor.Functor")

    def __call__(self, diagram):
        if isinstance(diagram, Bubble):
            return self(diagram.inside).map(diagram.func)
        if isinstance(diagram, monoidal.Sum):
            dom, cod = self(diagram.dom), self(diagram.cod)
            return sum(map(self, diagram), Tensor.zeros(dom, cod))
        if isinstance(diagram, monoidal.Ty):
            def obj_to_dim(obj):
                if isinstance(obj, rigid.Ob) and obj.z != 0:
                    obj = type(obj)(obj.name)  # sets z=0
                result = self.ob[type(diagram)(obj)]
                if isinstance(result, int):
                    result = Dim(result)
                if not isinstance(result, Dim):
                    result = Dim.upgrade(result)
                return result
            return Dim(1).tensor(*map(obj_to_dim, diagram.objects))
        if isinstance(diagram, Cup):
            return Tensor.cups(self(diagram.dom[:1]), self(diagram.dom[1:]))
        if isinstance(diagram, Cap):
            return Tensor.caps(self(diagram.cod[:1]), self(diagram.cod[1:]))
        if isinstance(diagram, monoidal.Box)\
                and not isinstance(diagram, monoidal.Swap):
            if diagram.z % 2 != 0:
                while diagram.z != 0:
                    diagram = diagram.l if diagram.z > 0 else diagram.r
                return self(diagram).conjugate()
            if diagram.is_dagger:
                return self(diagram.dagger()).dagger()
            return Tensor(self(diagram.dom), self(diagram.cod),
                          self.ar[diagram])
        if not isinstance(diagram, monoidal.Diagram):
            raise TypeError(messages.type_err(monoidal.Diagram, diagram))

        def dim(scan):
            return len(self(scan))
        scan, array = diagram.dom, Tensor.id(self(diagram.dom)).array
        for box, off in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, monoidal.Swap):
                source = range(
                    dim(diagram.dom @ scan[:off]),
                    dim(diagram.dom @ scan[:off] @ box.dom))
                target = [
                    i + dim(box.right)
                    if i < dim(diagram.dom @ scan[:off]) + dim(box.left)
                    else i - dim(box.left) for i in source]
                array = Tensor.np.moveaxis(array, list(source), list(target))
                scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
                continue
            left = dim(scan[:off])
            source = list(range(dim(diagram.dom) + left,
                                dim(diagram.dom) + left + dim(box.dom)))
            target = list(range(dim(box.dom)))
            array =\
                Tensor.np.tensordot(array, self(box).array, (source, target))
            source = range(len(array.shape) - dim(box.cod), len(array.shape))
            target = range(dim(diagram.dom) + left,
                           dim(diagram.dom) + left + dim(box.cod))
            array = Tensor.np.moveaxis(array, list(source), list(target))
            scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
        return Tensor(self(diagram.dom), self(diagram.cod), array)


@monoidal.Diagram.subclass
class Diagram(rigid.Diagram):
    """
    Diagram with Tensor boxes.

    Examples
    --------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> diagram = vector[::-1] >> vector @ vector
    >>> print(diagram)
    vector[::-1] >> vector >> Id(Dim(2)) @ vector
    """
    def eval(self, contractor=None):
        """
        Diagram evaluation.

        Parameters
        ----------
        contractor : callable, optional
            Use :class:`tensornetwork` contraction
            instead of :class:`tensor.Functor`.

        Returns
        -------
        tensor : Tensor
            With the same domain and codomain as self.

        Examples
        --------
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> assert (vector >> vector[::-1]).eval() == 1
        >>> import tensornetwork as tn
        >>> assert (vector >> vector[::-1]).eval(tn.contractors.auto) == 1
        """
        if contractor is None and "numpy" not in Tensor.np.__package__:
            raise Exception(
                'Please provide a contractor when using a non-numpy backend.')

        if contractor is None:
            return Functor(ob=lambda x: x, ar=lambda f: f.array)(self)
        array = contractor(*self.to_tn()).tensor
        return Tensor(self.dom, self.cod, array)

    def to_tn(self):
        """
        Sends a diagram to :code:`tensornetwork`.

        Returns
        -------
        nodes : :class:`tensornetwork.Node`
            Nodes of the network.

        output_edge_order : list of :class:`tensornetwork.Edge`
            Output edges of the network.

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
        nodes = [tn.Node(Tensor.np.eye(dim), 'input_{}'.format(i))
                 for i, dim in enumerate(self.dom)]
        inputs, scan = [n[0] for n in nodes], [n[1] for n in nodes]
        for box, offset in zip(self.boxes, self.offsets):
            if isinstance(box, Swap):
                scan[offset], scan[offset + 1] = scan[offset + 1], scan[offset]
                continue
            node = tn.Node(box.array, str(box))
            for i, _ in enumerate(box.dom):
                tn.connect(scan[offset + i], node[i])
            edges = [node[len(box.dom) + i] for i, _ in enumerate(box.cod)]
            scan = scan[:offset] + edges + scan[offset + len(box.dom):]
            nodes.append(node)
        return nodes, inputs + scan

    @staticmethod
    def cups(left, right):
        return rigid.cups(left, right, ar_factory=Diagram,
                          cup_factory=lambda x, _: Spider(2, 0, x[0]))

    @staticmethod
    def caps(left, right):
        return Diagram.cups(left, right).dagger()

    @staticmethod
    def swap(left, right):
        return monoidal.Diagram.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)

    def grad(self, var, **params):
        """ Gradient with respect to :code:`var`. """
        if var not in self.free_symbols:
            return self.sum([], self.dom, self.cod)
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
        Tensor(dom=Dim(1), cod=Dim(3, 2), array=[2.0*x, 0, 0, 1.0*z, 0, 1.0*y])
        """
        dim = Dim(len(variables) or 1)
        result = Sum([], self.dom, dim @ self.cod)
        for i, var in enumerate(variables):
            onehot = numpy.zeros(dim or (1, ))
            onehot[i] = 1
            result += Box(var, Dim(1), dim, onehot) @ self.grad(var)
        return result

    @staticmethod
    def spiders(n_legs_in, n_legs_out, dim):
        """
        Spider diagram.

        Parameters
        ----------
        n_legs_in, n_legs_out : int
            Number of legs in and out.
        dim : Dim
            Dimension for each leg.

        Examples
        --------
        >>> assert Diagram.spiders(3, 2, Dim(1)) == Id(Dim(1))
        >>> assert Diagram.spiders(1, 2, Dim(2)) == Spider(1, 2, Dim(2))
        >>> Diagram.spiders(1, 2, Dim(2, 3))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        NotImplementedError
        """
        dim = dim if isinstance(dim, Dim) else Dim(dim)
        if not dim:
            return Id(dim)
        if len(dim) == 1:
            return Spider(n_legs_in, n_legs_out, dim)
        raise NotImplementedError


class Id(rigid.Id, Diagram):
    """ Identity tensor.Diagram """
    def __init__(self, dom=Dim()):
        rigid.Id.__init__(self, dom)
        Diagram.__init__(self, dom, dom, [], [], layers=cat.Id(dom))


class Sum(monoidal.Sum, Diagram):
    """ Sums of tensor diagrams. """
    def eval(self, contractor=None):
        return sum(term.eval(contractor=contractor) for term in self.terms)


Diagram.id = Id
Diagram.sum = Sum


class Swap(rigid.Swap, Diagram):
    """ Symmetry in a tensor.Diagram """


class Box(rigid.Box, Diagram):
    """ Box in a tensor.Diagram """
    def __init__(self, name, dom, cod, data, **params):
        rigid.Box.__init__(self, name, dom, cod, data=data, **params)
        Diagram.__init__(self, dom, cod, [self], [0])

    @property
    def array(self):
        """ The array inside the box. """
        dom, cod = self.dom, self.cod
        return Tensor.np.array(self.data).reshape(tuple(dom @ cod) or (1, ))

    def grad(self, var, **params):
        return self.bubble(
            func=lambda x: getattr(x, "diff", lambda _: 0)(var),
            drawing_name="$\\partial {}$".format(var))

    def __repr__(self):
        return super().__repr__().replace("Box", "tensor.Box")

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return Tensor.np.all(self.array == other.array)\
            and (self.name, self.dom, self.cod)\
            == (other.name, other.dom, other.cod)

    def __hash__(self):
        return hash(
            (self.name, self.dom, self.cod, tuple(self.array.flatten())))


class Spider(rigid.Spider, Box):
    """
    Spider box.

    Parameters
    ----------
    n_legs_in, n_legs_out : int
        Number of legs in and out.
    dim : int
        Dimension for each leg.

    Examples
    --------
    >>> vector = Box('vec', Dim(1), Dim(2), [0, 1])
    >>> spider = Spider(1, 2, dim=2)
    >>> assert (vector >> spider).eval() == (vector @ vector).eval()
    >>> from discopy import drawing
    >>> drawing.equation(vector >> spider, vector @ vector, figsize=(3, 2),\\
    ... path='docs/_static/imgs/tensor/frobenius-example.png')

    .. image:: ../_static/imgs/tensor/frobenius-example.png
        :align: center
    """
    def __init__(self, n_legs_in, n_legs_out, dim):
        dim = dim if isinstance(dim, Dim) else Dim(dim)
        rigid.Spider.__init__(self, n_legs_in, n_legs_out, dim)
        array = numpy.zeros(self.dom @ self.cod)
        for i in range(int(numpy.prod(dim))):
            array[len(self.dom @ self.cod) * (i, )] = 1
        Box.__init__(self, self.name, self.dom, self.cod, array)
        self.dim = dim


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
        return Spider(1, 2, dim=self.dom)\
            >> self.inside.bubble(
                func=lambda x: self.func(tmp).diff(tmp).subs(tmp, x),
                drawing_name=name.format(self.drawing_name, var))\
            @ self.inside.grad(var) >> Spider(2, 1, dim=self.cod)


Diagram.bubble_factory = Bubble
