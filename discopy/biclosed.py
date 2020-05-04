from discopy import moncat


class Ty(moncat.Ty):
    """
    >>> x, y = Ty('x'), Ty('y')
    >>> print(y << x >> y)
    ((y << x) >> y)
    >>> print((y << x >> y) @ x)
    ((y << x) >> y) @ x
    """
    def __init__(self, name=None, left=None, right=None):
        self.left, self.right = left, right
        super().__init__(*(() if name is None else (name, )))

    def __lshift__(self, other):
        return Over(self, other)

    def __rshift__(self, other):
        return Under(self, other)


class Over(Ty):
    def __init__(self, left, right):
        super().__init__(self, left, right)

    def __repr__(self):
        return "Over({}, {})".format(repr(self.left), repr(self.right))

    def __str__(self):
        return "({} << {})".format(str(self.left), str(self.right))


class Under(Ty):
    def __init__(self, left, right):
        super().__init__(self, left, right)

    def __repr__(self):
        return "Under({}, {})".format(repr(self.left), repr(self.right))

    def __str__(self):
        return "({} >> {})".format(str(self.left), str(self.right))


class Diagram(moncat.Diagram):
    pass


class Box(moncat.Box, Diagram):
    pass


class Functor(moncat.Functor):
    """
    >>> from discopy import rigidcat
    >>> x, y = Ty('x'), Ty('y')
    >>> F = Functor(
    ...     ob={x: x, y: y}, ar={},
    ...     ob_factory=rigidcat.Ty,
    ...     ar_factory=rigidcat.Diagram)
    >>> print(F(y << x >> y))
    y.r @ x @ y.l
    >>> assert F((y << x) >> y) == F(y << (x >> y))
    """
    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Diagram):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __call__(self, diagram):
        if isinstance(diagram, Over):
            return self(diagram.left) >> self(diagram.right)
        if isinstance(diagram, Under):
            return self(diagram.left) << self(diagram.right)
        return super().__call__(diagram)
