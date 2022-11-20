from discopy import monoidal
from discopy.cat import factory
from discopy.utils import BinaryBoxConstructor


@factory
class Diagram(monoidal.Diagram):
    @staticmethod
    def swap(left, right, ar_factory=None, swap_factory=None):
        """
        Returns a diagram that swaps the left with the right wires.

        Parameters
        ----------
        left : monoidal.Ty
            left hand-side of the domain.
        right : monoidal.Ty
            right hand-side of the domain.

        Returns
        -------
        diagram : monoidal.Diagram
            with :code:`diagram.dom == left @ right`
        """
        ar_factory = ar_factory or Diagram
        swap_factory = swap_factory or Swap
        if not left:
            return ar_factory.id(right)
        if len(left) == 1:
            boxes = [
                swap_factory(left, right[i: i + 1])
                for i, _ in enumerate(right)]
            offsets = range(len(right))
            return ar_factory(left @ right, right @ left, boxes, offsets)
        return ar_factory.id(left[:1]) @ ar_factory.swap(left[1:], right)\
            >> ar_factory.swap(left[:1], right) @ ar_factory.id(left[1:])

    @staticmethod
    def permutation(perm, dom=None, ar_factory=None, inverse=False):
        """
        Returns the diagram that encodes a permutation of wires.

        .. warning::
            This method used to return the inverse permutation up to and
            including discopy v0.4.2.

        Parameters
        ----------
        perm : list of int
            such that :code:`i` goes to :code:`perm[i]`
        dom : monoidal.Ty, optional
            of the same length as :code:`perm`,
            default is :code:`PRO(len(perm))`.
        inverse : bool
            whether to return the inverse permutation.

        Returns
        -------
        diagram : monoidal.Diagram
        """
        ar_factory = ar_factory or Diagram
        if set(range(len(perm))) != set(perm):
            raise ValueError("Input should be a permutation of range(n).")
        if dom is None:
            dom = PRO(len(perm))
        if not inverse:
            warn_permutation.warn(
                'Since discopy v0.4.3 the behaviour of '
                'permutation has changed. Pass inverse=False '
                'to get the default behaviour.')
            perm = [perm.index(i) for i in range(len(perm))]
        if len(dom) != len(perm):
            raise ValueError(
                "Domain and permutation should have the same length.")
        diagram = ar_factory.id(dom)
        for i in range(len(dom)):
            j = perm.index(i)
            diagram = diagram >> ar_factory.id(diagram.cod[:i])\
                @ ar_factory.swap(diagram.cod[i:j], diagram.cod[j:j + 1])\
                @ ar_factory.id(diagram.cod[j + 1:])
            perm = perm[:i] + [i] + perm[i:j] + perm[j + 1:]
        return diagram

    def permute(self, *perm, inverse=False):
        """
        Returns :code:`self >> self.permutation(perm, self.dom)`.

        Parameters
        ----------
        perm : list of int
            such that :code:`i` goes to :code:`perm[i]`
        inverse : bool
            whether to return the inverse permutation.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 1, 0).cod == z @ y @ x
        >>> assert Id(x @ y @ z).permute(2, 0).cod == z @ y @ x
        """
        if min(perm) < 0 or max(perm) >= len(self.cod):
            raise IndexError(f'{self} index out of bounds.')
        if len(set(perm)) != len(perm):
            raise ValueError('{perm} is not a permutation.')
        sorted_perm = sorted(perm)
        perm = [
            i if i not in perm else sorted_perm[perm.index(i)]
            for i in range(len(self.cod))]
        return self >> self.permutation(list(perm), self.cod, inverse)


class Box(monoidal.Box, Diagram):
    pass


class Swap(BinaryBoxConstructor, Box):
    """
    Implements the symmetry of atomic types.

    Parameters
    ----------
    left : monoidal.Ty
        of length 1.
    right : monoidal.Ty
        of length 1.
    """
    def __init__(self, left, right):
        if len(left) != 1 or len(right) != 1:
            raise ValueError(messages.swap_vs_swaps(left, right))
        name, dom, cod =\
            "Swap({}, {})".format(left, right), left @ right, right @ left
        BinaryBoxConstructor.__init__(self, left, right)
        Box.__init__(self, name, dom, cod)
        self.draw_as_wires = True

    def __repr__(self):
        return "Swap({}, {})".format(repr(self.left), repr(self.right))

    def dagger(self):
        return type(self)(self.right, self.left)
