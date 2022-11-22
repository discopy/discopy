# -*- coding: utf-8 -*-

""" Spiders, i.e. dagger special commutative Frobenius algebras. """

from discopy.rigid import Box, Diagram


class Spider(Box):
    """
    Spider box.

    Parameters
    ----------
    n_legs_in, n_legs_out : int
        Number of legs in and out.
    typ : discopy.rigid.Ty
        The type of the spider, needs to be atomic.

    Examples
    --------
    >>> x = Ty('x')
    >>> spider = Spider(1, 2, x)
    >>> assert spider.dom == x and spider.cod == x @ x
    """
    def __init__(self, n_legs_in, n_legs_out, typ, **params):
        self.typ = typ
        if len(typ) > 1:
            raise ValueError(
                "Spider boxes can only have len(typ) == 1, "
                "try Diagram.spiders instead.")
        name = "Spider({}, {}, {})".format(n_legs_in, n_legs_out, typ)
        dom, cod = typ ** n_legs_in, typ ** n_legs_out
        cup_like = (n_legs_in, n_legs_out) in ((2, 0), (0, 2))
        params = dict(dict(
            draw_as_spider=not cup_like,
            draw_as_wires=cup_like,
            color="black", drawing_name=""), **params)
        Box.__init__(self, name, dom, cod, **params)

    def __repr__(self):
        return "Spider({}, {}, {})".format(
            len(self.dom), len(self.cod), repr(self.typ))

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), self.typ)

    def decompose(self):
        return self._decompose_spiders(len(self.dom), len(self.cod),
                                       self.typ)

    @classmethod
    def _decompose_spiders(cls, n_legs_in, n_legs_out, typ):
        if n_legs_out > n_legs_in:
            return cls._decompose_spiders(n_legs_out, n_legs_in,
                                          typ).dagger()

        if n_legs_in == 1 and n_legs_out == 0:
            return cls(1, 0, typ)
        if n_legs_in == 1 and n_legs_out == 1:
            return Id(typ)

        if n_legs_out != 1:
            return (cls._decompose_spiders(n_legs_in, 1, typ)
                    >> cls._decompose_spiders(1, n_legs_out, typ))

        if n_legs_in == 2:
            return cls(2, 1, typ)

        if n_legs_in % 2 == 1:
            return (cls._decompose_spiders(n_legs_in - 1, 1, typ)
                    @ Id(typ) >> cls(2, 1, typ))

        new_in = n_legs_in // 2
        half_spider = cls._decompose_spiders(new_in, 1, typ)
        return half_spider @ half_spider >> cls(2, 1, typ)

    @property
    def l(self):
        return type(self)(len(self.dom), len(self.cod), self.typ.l)

    @property
    def r(self):
        return type(self)(len(self.dom), len(self.cod), self.typ.r)


def spiders(
        n_legs_in, n_legs_out, typ,
        ar_factory=Diagram, spider_factory=Spider):
    """ Constructs a diagram of interleaving spiders. """
    id, swap, spider = ar_factory.id, ar_factory.swap, spider_factory
    ts = [typ[i:i + 1] for i in range(len(typ))]
    result = id().tensor(*[spider(n_legs_in, n_legs_out, t) for t in ts])

    for i, t in enumerate(ts):
        for j in range(n_legs_in - 1):
            result <<= id(result.dom[:i * j + i + j]) @ swap(
                t, result.dom[i * j + i + j:i * n_legs_in + j]
            ) @ id(result.dom[i * n_legs_in + j + 1:])

        for j in range(n_legs_out - 1):
            result >>= id(result.cod[:i * j + i + j]) @ swap(
                result.cod[i * j + i + j:i * n_legs_out + j], t
            ) @ id(result.cod[i * n_legs_out + j + 1:])
    return result


def coherence(factory):
    def method(cls, a: int, b: int, x: Ty, phase=None) -> Diagram:
        if len(x) == 0 and phase is None:
            return cls.id(x)
        if len(x) == 1:
            return factory(a, b, x, phase)
        if phase is not None:  # Coherence for phase shifters.
            shift = cls.tensor(*[factory(1, 1, obj, phase) for obj in x])
            return method(cls, a, 1, x) >> shift >> method(cls, 1, b, x)
        if (a, b) in [(1, 0), (0, 1)]: # Coherence for (co)units.
            return cls.tensor(*[factory(a, b, obj) for obj in x])
        # Coherence for binary (co)products.
        if (a, b) in [(1, 2), (2, 1)]:
            spiders, braids = (
                factory(a, b, x[0], phase) @ method(cls, a, b, x[1:], phase),
                x[0] @ cls.braid(x[0], x[1:]) @ x[1:])
            return spiders >> braids if (a, b) == (1, 2) else braids >> spiders
        if a == 1:  # We can now assume b > 2.
            return method(cls, 1, b - 1, x)\
                >> method(cls, 1, 2, x) @ (x ** (b - 2))
        if b == 1:  # We can now assume a > 2.
            return method(cls, 2, 1, x) @ (x ** (a - 2))\
                >> method(cls, a - 1, 1, x)
        return method(cls, a, 1, x) >> method(cls, 1, b, x)

    return classmethod(method)
