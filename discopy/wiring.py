# -*- coding: utf-8 -*-

"""
Implements wiring diagrams as a free dagger PROP.
"""

from abc import ABC, abstractmethod
import functools
import itertools

from discopy import cat, messages, monoidal
from discopy.monoidal import PRO, Ty

class Wiring(ABC, monoidal.Box):
    """
    Implements wiring diagrams in free dagger PROPs.
    """

    @abstractmethod
    def collapse(self, falg):
        """
        Collapse a wiring diagram catamorphically into a single domain,
        codomain, and auxiliary data item.
        """

    @staticmethod
    def id(dom):
        return Id(dom)

    def then(self, other):
        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))
        if isinstance(other, Id):
            return self
        return Sequential([self, other])

    def tensor(self, other):
        if isinstance(other, Id) and not other.dom:
            return self
        return Parallel([self, other])

    def __matmul__(self, other):
        return self.tensor(other)

    @abstractmethod
    def dagger(self):
        pass

class Id(Wiring):
    """ Empty wiring diagram in a free dagger PROP. """
    def __init__(self, dom):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        super().__init__("Id(dom={})".format(dom), dom, dom)

    def __repr__(self):
        return "Id(dom={})".format(repr(self.dom))

    def collapse(self, falg):
        return falg(self)

    def then(self, other):
        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))
        return other

    def tensor(self, other):
        if not self.dom:
            return other
        if isinstance(other, Id):
            return Id(self.dom @ other.dom)
        return super().tensor(other)

    def dagger(self):
        return Id(self.dom)

class Box(Wiring):
    """ Implements boxes in wiring diagrams. """
    def __init__(self, name, dom, cod, **params):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        super().__init__(name, dom, cod, **params)

    def __repr__(self):
        return "Box({}, dom={}, cod={}, data={})".format(
            repr(self.name), repr(self.dom), repr(self.cod), repr(self.data)
        )

    def collapse(self, falg):
        return falg(self)

    def dagger(self):
        if self.name[-1] == '†':
            name = self.name[:-1]
        else:
            name = self.name + '†'
        return Box(name, self.cod, self.dom, data=self.data)

def _flatten_arrows(arrows):
    for arr in arrows:
        if isinstance(arr, Id):
            continue
        elif isinstance(arr, Sequential):
            yield arr.arrows
        else:
            yield [arr]

class Sequential(Wiring):
    """ Sequential composition in a wiring diagram. """
    def __init__(self, arrows, dom=None, cod=None):
        arrows = list(itertools.chain(*_flatten_arrows(arrows)))
        for f, g in zip(arrows, arrows[1:]):
            if isinstance(f, Wiring) and isinstance(g, Wiring) and\
               f.cod != g.dom:
                raise cat.AxiomError(messages.does_not_compose(f, g))
        self.arrows = arrows

        if dom is None:
            dom = self.arrows[0].dom
        if cod is None:
            cod = self.arrows[-1].cod
        super().__init__(repr(self), dom, cod)

    def __repr__(self):
        return "Sequential(arrows={})".format(repr(self.arrows))

    def collapse(self, falg):
        return falg(functools.reduce(lambda f, g: f >> g,
                                     [f.collapse(falg) for f in self.arrows]))

    def dagger(self):
        return Sequential(reversed([f.dagger() for f in self.arrows]))

def _flatten_factors(factors):
    for f in factors:
        if isinstance(f, Id) and not f.dom:
            continue
        if isinstance(f, Parallel):
            yield f.factors
        else:
            yield [f]

class Parallel(Wiring):
    """ Parallel composition in a wiring diagram. """
    def __init__(self, factors, dom=None, cod=None):
        self.factors = list(itertools.chain(*_flatten_factors(factors)))
        if dom is None:
            dom = functools.reduce(lambda f, g: f @ g,
                                   (f.dom for f in self.factors), Ty())
        if cod is None:
            cod = functools.reduce(lambda f, g: f @ g,
                                   (f.cod for f in self.factors), Ty())
        super().__init__(repr(self), dom, cod)

    def __repr__(self):
        return "Parallel(factors={})".format(repr(self.factors))

    def collapse(self, falg):
        return falg(functools.reduce(lambda f, g: f @ g,
                                     [f.collapse(falg) for f in self.factors]))

    def dagger(self):
        return Parallel([f.dagger() for f in self.factors])

class Functor(monoidal.Functor):
    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Box):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __functor_falg__(self, f):
        if isinstance(f, Id):
            return self.ar_factory.id(f.dom)
        if isinstance(f, Box):
            return self.ar[f]
        if isinstance(f, Sequential):
            return functools.reduce(lambda a, b: a >> b, f.arrows)
        if isinstance(f, Parallel):
            return functools.reduce(lambda a, b: a @ b, f.factors)
        return f

    def __call__(self, diagram):
        if isinstance(diagram, Wiring):
            return diagram.collapse(self.__functor_falg__)
        return super().__call__(diagram)

class WiringFunctor(Functor):
    def __init__(self):
        ob = lambda t: PRO(len(t))
        ar = lambda f: Box(f.name, PRO(len(f.dom)), PRO(len(f.cod)),
                           data=f.data)
        super().__init__(ob, ar, ob_factory=PRO, ar_factory=Box)
