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

from discopy import messages, monoidal, rigid, config
from discopy.cat import AxiomError
from discopy.monoidal import Sum
from discopy.rigid import Ob, Ty, Cup, Cap


if config.IMPORT_JAX:  # pragma: no cover
    import warnings
    for msg in config.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore", message=msg)
    import jax.numpy as np

    def array2string(array, max_length=config.NUMPY_THRESHOLD):
        """ array2string is not implemented in jax.numpy """
        flat = list(array)
        flat = flat if len(flat) <= max_length else\
            flat[:max_length // 2] + ["..."] + flat[1 - max_length // 2:]
        return "[{}]".format(", ".join(map(str, flat)))
    np.array2string = array2string
else:
    import numpy as np
    from numpy import array2string as _array2string
    np.set_printoptions(threshold=config.NUMPY_THRESHOLD)

    def array2string(array, **params):
        """ makes sure we get the same doctest with numpy and jax.numpy """
        return _array2string(array, **dict(params, separator=', '))\
            .replace('[ ', '[').replace('  ', ' ')
    np.array2string = array2string


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
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError(messages.type_err(int, dim))
            if dim < 1:
                raise ValueError
        super().__init__(*[Ob(dim) for dim in dims if dim > 1])

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

    >>> m = Tensor(Dim(2), Dim(2), [0, 1, 1, 0])
    >>> v = Tensor(Dim(1), Dim(2), [0, 1])
    >>> v >> m >> v.dagger()
    Tensor(dom=Dim(1), cod=Dim(1), array=[0])
    """
    def __init__(self, dom, cod, array):
        self._array = np.array(array).reshape(dom @ cod or (1, ))
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
            self.dom, self.cod,
            np.array2string(self.array.flatten()))

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
            return np.all(self.array == other)
        return (self.dom, self.cod) == (other.dom, other.cod)\
            and np.all(self.array == other.array)

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other, = others
        if not isinstance(other, Tensor):
            raise TypeError(messages.type_err(Tensor, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        array = np.tensordot(self.array, other.array, len(self.cod))\
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
        array = np.tensordot(self.array, other.array, 0)\
            if self.array.shape and other.array.shape\
            else self.array * other.array
        source = range(len(dom @ cod))
        target = [
            i if i < len(self.dom) or i >= len(self.dom @ self.cod @ other.dom)
            else i - len(self.cod) if i >= len(self.dom @ self.cod)
            else i + len(other.dom) for i in source]
        return Tensor(dom, cod, np.moveaxis(array, source, target))

    def dagger(self):
        array = np.moveaxis(
            self.array, range(len(self.dom @ self.cod)),
            [i + len(self.cod) if i < len(self.dom) else
             i - len(self.dom) for i in range(len(self.dom @ self.cod))])
        return Tensor(self.cod, self.dom, np.conjugate(array))

    @staticmethod
    def id(dom):
        return Tensor(dom, dom, np.identity(int(np.prod(dom))))

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
                      np.moveaxis(array, source, target))

    def transpose(self, left=False):
        """
        Returns the algebraic transpose.

        Note
        ----
        This is *not* the same as the diagrammatic transpose for complex dims.
        """
        return Tensor(self.cod[::-1], self.dom[::-1], self.array.transpose())

    def conjugate(self):
        """ Returns the conjugate of a tensor. """
        return Tensor(self.dom, self.cod, np.conjugate(self.array))

    def round(self, decimals=0):
        """ Rounds the entries of a tensor up to a number of decimals. """
        return Tensor(self.dom, self.cod,
                      np.around(self.array, decimals=decimals))

    @staticmethod
    def zeros(dom, cod):
        """
        Returns the zero tensor of a given shape.

        Examples
        --------
        >>> assert Tensor.zeros(Dim(2), Dim(2))\\
        ...     == Tensor(Dim(2), Dim(2), [0, 0, 0, 0])
        """
        return Tensor(dom, cod, np.zeros(dom @ cod))

    def subs(self, *args):
        array = [getattr(x, "subs", lambda y, *_: y)(*args)
                 for x in self.array.flatten()]
        return Tensor(self.dom, self.cod, array)

    def grad(self, *variables):  # pragma: no cover
        """ Gradient with respect to vars. """
        array = np.array([[
            getattr(x, "diff", lambda _: 0)(var) for x in self.array.flatten()]
            for var in variables])
        array = array.reshape(Dim(len(variables)) @ self.dom @ self.cod)
        return Tensor(Dim(len(variables)) @ self.dom, self.cod, array)


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
        if isinstance(diagram, monoidal.Sum):
            dom, cod = self(diagram.dom), self(diagram.cod)
            return sum(map(self, diagram), Tensor.zeros(dom, cod))
        if isinstance(diagram, monoidal.Ty):
            def obj_to_dim(obj):
                result = self.ob[type(diagram)(obj.name)]
                return result if isinstance(result, Dim) else Dim(result)
            return Dim(1).tensor(*map(obj_to_dim, diagram.objects))
        if isinstance(diagram, Cup):
            return Tensor.cups(self(diagram.dom[:1]), self(diagram.dom[1:]))
        if isinstance(diagram, Cap):
            return Tensor.caps(self(diagram.cod[:1]), self(diagram.cod[1:]))
        if isinstance(diagram, monoidal.Box)\
                and not isinstance(diagram, monoidal.Swap):
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
                array = np.moveaxis(array, list(source), list(target))
                scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
                continue
            left = dim(scan[:off])
            source = list(range(dim(diagram.dom) + left,
                                dim(diagram.dom) + left + dim(box.dom)))
            target = list(range(dim(box.dom)))
            array = np.tensordot(array, self(box).array, (source, target))
            source = range(len(array.shape) - dim(box.cod), len(array.shape))
            target = range(dim(diagram.dom) + left,
                           dim(diagram.dom) + left + dim(box.cod))
            array = np.moveaxis(array, list(source), list(target))
            scan = scan[:off] @ box.cod @ scan[off + len(box.dom):]
        return Tensor(self(diagram.dom), self(diagram.cod), array)


@monoidal.diagram_subclass
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
        >>> from tensornetwork import Node, Edge
        >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
        >>> nodes, output_edge_order = vector.to_tn()
        >>> node, = nodes
        >>> assert node.name == "vector" and np.all(node.tensor == [0, 1])
        >>> assert output_edge_order == [node[0]]
        """
        import tensornetwork as tn
        nodes = [tn.Node(np.eye(dim), 'input_{}'.format(i))
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
                          cup_factory=lambda x, _: Frobenius(2, 0, x[0]))

    @staticmethod
    def caps(left, right):
        return Diagram.cups(left, right).dagger()

    @staticmethod
    def swap(left, right):
        return monoidal.swap(
            left, right, ar_factory=Diagram, swap_factory=Swap)


class Id(rigid.Id, Diagram):
    """ Identity tensor.Diagram """


Diagram.id = Id


class Swap(rigid.Swap, Diagram):
    """ Symmetry in a tensor.Diagram """


class Box(rigid.Box, Diagram):
    """ Box in a tensor.Diagram """
    @property
    def array(self):
        """ The array inside the box. """
        return np.array(self.data).reshape(self.dom @ self.cod or (1, ))

    def __repr__(self):
        return super().__repr__().replace("Box", "tensor.Box")

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return np.all(self.array == other.array)\
            and (self.name, self.dom, self.cod)\
            == (other.name, other.dom, other.cod)

    def __hash__(self):
        return hash(
            (self.name, self.dom, self.cod, tuple(self.array.flatten())))


class Frobenius(Box):
    """
    Frobenius box.

    Parameters
    ----------
    n_wires_in, n_wires_out : int
        Number of input and output wires.
    dim : int
        Dimension for each leg.

    Examples
    --------
    >>> vector = Box('vector', Dim(1), Dim(2), [0, 1])
    >>> spider = Frobenius(1, 2, dim=2)
    >>> assert (vector >> spider).eval() == (vector @ vector).eval()
    """
    def __init__(self, n_wires_in, n_wires_out, dim):
        import numpy
        name = "Frobenius({}, {}, dim={})".format(n_wires_in, n_wires_out, dim)
        dom, cod = Dim(dim) ** n_wires_in, Dim(dim) ** n_wires_out
        array = numpy.zeros(dom @ cod)
        for i in range(dim):
            array[len(dom @ cod) * (i, )] = 1
        self.draw_as_spider, self.color, self.drawing_name = True, "black", ""
        self.dim = dim
        super().__init__(name, dom, cod, array)

    def __repr__(self):
        return self.name

    def dagger(self):
        return Frobenius(len(self.cod), len(self.dom), self.dim)
