# -*- coding: utf-8 -*-

""" DisCoPy rewriting methods. """


from discopy import cat, messages


def interchange(self, i, j, left=False):
    """
    Returns a new diagram with boxes i and j interchanged.

    Gets called recursively whenever :code:`i < j + 1 or j < i - 1`.

    Parameters
    ----------
    i : int
        Index of the box to interchange.
    j : int
        Index of the new position for the box.
    left : bool, optional
        Whether to apply left interchangers.

    Notes
    -----
    By default, we apply only right exchange moves::

        top >> Id(left @ box1.dom @ mid) @ box0 @ Id(right)
            >> Id(left) @ box1 @ Id(mid @ box0.cod @ right) >> bottom

    gets rewritten to::

        top >> Id(left) @ box1 @ Id(mid @ box0.dom @ right)
            >> Id(left @ box1.cod @ mid) @ box0 @ Id(right) >> bottom
    """
    from discopy.monoidal import Layer, Diagram
    if not 0 <= i < len(self) or not 0 <= j < len(self):
        raise IndexError
    if i == j:
        return self
    if j < i - 1:
        result = self
        for k in range(i - j):
            result = result.interchange(i - k, i - k - 1, left=left)
        return result
    if j > i + 1:
        result = self
        for k in range(j - i):
            result = result.interchange(i + k, i + k + 1, left=left)
        return result
    if j < i:
        i, j = j, i
    off0, off1 = self.offsets[i], self.offsets[j]
    left0, box0, right0 = self.layers[i]
    left1, box1, right1 = self.layers[j]
    # By default, we check if box0 is to the right first, then to the left.
    if left and off1 >= off0 + len(box0.cod):  # box0 left of box1
        off1 = off1 - len(box0.cod) + len(box0.dom)
        middle = left1[len(left0 @ box0.cod):]
        layer0 = self.layer_factory(left0, box0, middle @ box1.cod @ right1)
        layer1 = self.layer_factory(left0 @ box0.dom @ middle, box1, right1)
    elif off0 >= off1 + len(box1.dom):  # box0 right of box1
        off0 = off0 - len(box1.dom) + len(box1.cod)
        middle = left0[len(left1 @ box1.dom):]
        layer0 = self.layer_factory(left1 @ box1.cod @ middle, box0, right0)
        layer1 = self.layer_factory(left1, box1, middle @ box0.dom @ right0)
    elif off1 >= off0 + len(box0.cod):  # box0 left of box1
        off1 = off1 - len(box0.cod) + len(box0.dom)
        middle = left1[len(left0 @ box0.cod):]
        layer0 = self.layer_factory(left0, box0, middle @ box1.cod @ right1)
        layer1 = self.layer_factory(left0 @ box0.dom @ middle, box1, right1)
    else:
        raise InterchangerError(box0, box1)
    boxes = self.boxes[:i] + [box1, box0] + self.boxes[i + 2:]
    offsets = self.offsets[:i] + [off1, off0] + self.offsets[i + 2:]
    layers = self.layers[:i] >> layer1 >> layer0 >> self.layers[i + 2:]
    return self.upgrade(
        Diagram(self.dom, self.cod, boxes, offsets, layers=layers))


class InterchangerError(cat.AxiomError):
    """ This is raised when we try to interchange conected boxes. """
    def __init__(self, box0, box1):
        super().__init__("Boxes {} and {} do not commute.".format(box0, box1))


def normalize(self, left=False):
    """
    Implements normalisation of connected diagrams, see arXiv:1804.07832.

    Parameters
    ----------
    left : bool, optional
        Passed to :meth:`Diagram.interchange`.

    Yields
    ------
    diagram : :class:`Diagram`
        Rewrite steps.

    Examples
    --------

    >>> from discopy.monoidal import *
    >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
    >>> gen = (s0 @ s1).normalize()
    >>> for _ in range(3): print(next(gen))
    s1 >> s0
    s0 >> s1
    s1 >> s0
    """
    diagram = self
    while True:
        no_more_moves = True
        for i in range(len(diagram) - 1):
            box0, box1 = diagram.boxes[i], diagram.boxes[i + 1]
            off0, off1 = diagram.offsets[i], diagram.offsets[i + 1]
            if left and off1 >= off0 + len(box0.cod)\
                    or not left and off0 >= off1 + len(box1.dom):
                diagram = diagram.interchange(i, i + 1, left=left)
                yield diagram
                no_more_moves = False
        if no_more_moves:
            break


def normal_form(self, normalizer=None, **params):
    """
    Returns the normal form of a diagram.

    Parameters
    ----------
    normalizer : iterable of :class:`Diagram`, optional
        Generator that yields rewrite steps, default is
        :meth:`Diagram.normalize`.

    params : any, optional
        Passed to :code:`normalizer`.

    Raises
    ------
    NotImplementedError
        Whenever :code:`normalizer` yields the same rewrite steps twice.
    """
    from discopy.monoidal import Diagram
    diagram, cache = self, set()
    for _diagram in (normalizer or Diagram.normalize)(diagram, **params):
        if _diagram in cache:
            raise NotImplementedError(messages.is_not_connected(self))
        diagram = _diagram
        cache.add(diagram)
    return diagram


def foliate(self, yield_slices=False):
    """
    Generator yielding the interchanger steps in the foliation of self.

    Yields
    ------
    diagram : :class:`Diagram`
        Rewrite steps of the foliation.

    Parameters
    ----------
    yield_slices : bool, optional
        Yield the list of slices of self as last output,
        used in :meth:`Diagram.foliation`.

    Examples
    --------

    >>> from discopy.monoidal import *
    >>> x, y = Ty('x'), Ty('y')
    >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
    >>> d = (f0 @ Id(x) >> f0.dagger() @ f1.dagger()) @ (f0 >> f1)
    >>> *_, slices = d.foliate(yield_slices=True)
    >>> print(slices[0])
    f0 @ Id(x @ x) >> Id(y) @ f1[::-1] @ Id(x) >> Id(y @ y) @ f0
    >>> print(slices[1])
    f0[::-1] @ Id(y @ y) >> Id(x @ y) @ f1

    >>> d.draw(figsize=(4, 2),
    ...        path='docs/_static/imgs/monoidal/foliate-example-1a.png')

    .. image:: ../_static/imgs/monoidal/foliate-example-1a.png
        :align: center

    >>> drawing.equation(
    ...     *slices, symbol=', ', figsize=(4, 2),
    ...     path='docs/_static/imgs/monoidal/foliate-example-1b.png')

    .. image:: ../_static/imgs/monoidal/foliate-example-1b.png
        :align: center

    >>> ket = Box('ket', Ty(), x)
    >>> scalar = Box('scalar', Ty(), Ty())
    >>> kets = scalar @ ket @ scalar @ ket
    >>> a = kets.foliate()
    >>> assert next(a) == kets

    >>> kets.draw(figsize=(2, 2),
    ...           path='docs/_static/imgs/monoidal/foliate-example-2.png')

    .. image:: ../_static/imgs/monoidal/foliate-example-2.png
        :align: center

    """
    def is_right_of(last, diagram):
        off0, off1 = diagram.offsets[last], diagram.offsets[last + 1]
        box0, box1 = diagram.boxes[last], diagram.boxes[last + 1]
        if off1 >= off0 + len(box0.cod):  # box1 right of box0
            return True
        if off0 >= off1 + len(box1.dom):  # box1 left of box0
            return False
        return None

    def move_in_slice(first, last, k, diagram):
        result = diagram
        try:
            if not k == last + 1:
                result = diagram.interchange(k, last + 1)
            right_of_last = is_right_of(last, result)
            if right_of_last is None:
                return None
            if right_of_last:
                return result
            result = result.interchange(last + 1, last)
            if last == first:
                return result
            return move_in_slice(first, last - 1, last, result)
        except InterchangerError:
            return None

    start, diagram = 0, self
    if yield_slices:
        slices = []
    while start < len(diagram):
        last = start
        k = last + 1
        while k < len(diagram):
            result = move_in_slice(start, last, k, diagram)
            k += 1
            if result is None:
                pass
            else:
                diagram = result
                last += 1
                yield diagram
        if yield_slices:
            slices += [diagram[start: last + 1]]
        start = last + 1
    if yield_slices:
        yield slices


def flatten(self):
    """
    Takes a diagram of diagrams and returns a diagram.

    >>> from discopy.monoidal import *
    >>> x, y = Ty('x'), Ty('y')
    >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
    >>> g = Box('g', x @ y, y)
    >>> d = (Id(y) @ f0 @ Id(x) >> f0.dagger() @ Id(y) @ f0 >>\\
    ...      g @ f1 >> f1 @ Id(x)).normal_form()
    >>> assert d.foliation().flatten().normal_form() == d
    >>> assert d.foliation().dagger().flatten()\\
    ...     == d.foliation().flatten().dagger()
    """
    from discopy.monoidal import Functor
    return self.upgrade(Functor(lambda x: x, lambda f: f)(self))


def foliation(self):
    """
    Returns a diagram with normal_form diagrams of depth 1 as boxes
    such that its flattening gives the original diagram back.

    >>> from discopy.monoidal import *
    >>> x, y = Ty('x'), Ty('y')
    >>> f0, f1 = Box('f0', x, y), Box('f1', y, x)
    >>> d = f0 @ Id(y) >> f0.dagger() @ f1
    >>> assert d.foliation().boxes[0] == f0 @ f1
    >>> assert d.foliation().flatten().normal_form() == d
    >>> assert d.foliation().flatten()\\
    ...     == d[::-1].foliation()[::-1].flatten()\\
    ...     == d[::-1].foliation().flatten()[::-1]
    >>> assert d.foliation().flatten().foliation() == d.foliation()
    >>> g = Box('g', x @ x, x @ y)
    >>> diagram = (d >> g >> d) @ (d >> g >> d)
    >>> slices = diagram.foliation()
    >>> assert slices.boxes[0] == f0 @ f1 @ f0 @ f1
    >>> *_, last_diagram = diagram.foliate()
    >>> assert last_diagram == slices.flatten()
    """
    from discopy.monoidal import Diagram
    *_, slices = self.foliate(yield_slices=True)
    return self.upgrade(
        Diagram(self.dom, self.cod, slices, len(slices) * [0]))


def depth(self):
    """
    Computes the depth of a diagram by foliating it.

    >>> from discopy.monoidal import *
    >>> x, y = Ty('x'), Ty('y')
    >>> f, g = Box('f', x, y), Box('g', y, x)
    >>> assert Id(x @ y).depth() == 0
    >>> assert f.depth() == 1
    >>> assert (f @ g).depth() == 1
    >>> assert (f >> g).depth() == 2
    """
    *_, slices = self.foliate(yield_slices=True)
    return len(slices)


def width(self):
    """
    Computes the width of a diagram,
    i.e. the maximum number of parallel wires.

    >>> from discopy.monoidal import *
    >>> x = Ty('x')
    >>> f = Box('f', x, x ** 4)
    >>> assert (f @ Id(x ** 2) >> Id(x ** 2) @ f.dagger()).width() == 6
    """
    return max(len(self.dom), max(
        len(left @ box.cod @ right) for left, box, right in self.layers))


def snake_removal(self, left=False):
    """
    Return a generator which yields normalization steps.

    >>> from discopy.rigid import *
    >>> n, s = Ty('n'), Ty('s')
    >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
    >>> f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    >>> diagram = g @ cap >> f[::-1] @ Id(n.r) @ f >> cup @ h
    >>> for d in diagram.normalize(): print(d)  # doctest: +ELLIPSIS
    g... >> Cup(n, n.r) @ Id(n)...
    g >> f[::-1] >> Id(n) @ Cap(n.r, n) >> Cup(n, n.r) @ Id(n) >> f >> h
    g >> f[::-1] >> f >> h
    """
    from discopy import monoidal
    from discopy.rigid import Diagram, Cup, Cap

    def follow_wire(diagram, i, j):
        """
        Given a diagram, the index of a box i and the offset j of an output
        wire, returns (i, j, obstructions) where:
        - i is the index of the box which takes this wire as input, or
        len(diagram) if it is connected to the bottom boundary.
        - j is the offset of the wire at its bottom end.
        - obstructions is a pair of lists of indices for the boxes on
        the left and right of the wire we followed.
        """
        left_obstruction, right_obstruction = [], []
        while i < len(diagram) - 1:
            i += 1
            box, off = diagram.boxes[i], diagram.offsets[i]
            if off <= j < off + len(box.dom):
                return i, j, (left_obstruction, right_obstruction)
            if off <= j:
                j += len(box.cod) - len(box.dom)
                left_obstruction.append(i)
            else:
                right_obstruction.append(i)
        return len(diagram), j, (left_obstruction, right_obstruction)

    def find_snake(diagram):
        """
        Given a diagram, returns (cup, cap, obstructions, left_snake)
        if there is a yankable pair, otherwise returns None.
        """
        for cap in range(len(diagram)):
            if not isinstance(diagram.boxes[cap], Cap):
                continue
            for left_snake, wire in [(True, diagram.offsets[cap]),
                                     (False, diagram.offsets[cap] + 1)]:
                cup, wire, obstructions =\
                    follow_wire(diagram, cap, wire)
                not_yankable =\
                    cup == len(diagram)\
                    or not isinstance(diagram.boxes[cup], Cup)\
                    or left_snake and diagram.offsets[cup] + 1 != wire\
                    or not left_snake and diagram.offsets[cup] != wire
                if not_yankable:
                    continue
                return cup, cap, obstructions, left_snake
        return None

    def unsnake(diagram, cup, cap, obstructions, left_snake=False):
        """
        Given a diagram and the indices for a cup and cap pair
        and a pair of lists of obstructions on the left and right,
        returns a new diagram with the snake removed.

        A left snake is one of the form Id @ Cap >> Cup @ Id.
        A right snake is one of the form Cap @ Id >> Id @ Cup.
        """
        left_obstruction, right_obstruction = obstructions
        if left_snake:
            for box in left_obstruction:
                diagram = diagram.interchange(box, cap)
                yield diagram
                for i, right_box in enumerate(right_obstruction):
                    if right_box < box:
                        right_obstruction[i] += 1
                cap += 1
            for box in right_obstruction[::-1]:
                diagram = diagram.interchange(box, cup)
                yield diagram
                cup -= 1
        else:
            for box in left_obstruction[::-1]:
                diagram = diagram.interchange(box, cup)
                yield diagram
                for i, right_box in enumerate(right_obstruction):
                    if right_box > box:
                        right_obstruction[i] -= 1
                cup -= 1
            for box in right_obstruction:
                diagram = diagram.interchange(box, cap)
                yield diagram
                cap += 1
        boxes = diagram.boxes[:cap] + diagram.boxes[cup + 1:]
        offsets = diagram.offsets[:cap] + diagram.offsets[cup + 1:]
        layers = diagram.layers[:cap] >> diagram.layers[cup + 1:]
        yield Diagram(diagram.dom, diagram.cod, boxes, offsets, layers)

    diagram = self
    while True:
        yankable = find_snake(diagram)
        if yankable is None:
            break
        for _diagram in unsnake(diagram, *yankable):
            yield _diagram
            diagram = _diagram
    for _diagram in monoidal.Diagram.normalize(diagram, left=left):
        yield _diagram
