# -*- coding: utf-8 -*-

"""
Token passing for linear lambda terms.

A linear lambda term (:class:`discopy.closed.Term`) compiles to a string
diagram whose boxes are evaluations (applications, ``Eval``) and coevaluations
(abstractions, ``Coeval``). This module gives that diagram a *token-passing*
semantics in the additive category of Python functions
(:mod:`discopy.python.additive`), in the spirit of the Geometry of Interaction:
a token walks the diagram carrying a **stack of bits** recording which
*premise* of each application/abstraction it came from, and the bits pushed
at an abstraction route the token through the application it meets, and vice
versa (see :func:`token_function`).

The machine is assembled as a :class:`discopy.python.additive.Hypergraph`,
which is *right-monogamous* (each wire has a unique consumer) and need not be
causal -- the shape of a token machine, where each undirected wire of the term
becomes a pair of opposite directed wires the token can travel along. Calling
the hypergraph runs the machine, e.g. the token for ``(lambda z: z)(x)``
enters on the root wire and exits at the free variable ``x`` with a balanced
stack. The term itself is recovered by :func:`to_term`, so we get a round trip

    term -> additive hypergraph -> term

up to alpha-equivalence (see :func:`roundtrip`).

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    token_function
    to_hypergraph
    to_term
    roundtrip
"""

from __future__ import annotations

from discopy.cmap import PortKind
from discopy.python.additive import Function, Hypergraph


def token_function(box) -> Function:
    """
    The token-passing :class:`Function` of a box in a lambda diagram.

    The box has ``len(box.dom) + len(box.cod)`` wires meeting it, each both an
    entry (its domain) and an exit (its codomain). An application (``Eval``)
    is a tensor node with the function wire as *conclusion* and the argument
    and result wires as *premises*; an abstraction (``Coeval``) is the dual
    par node with the context wire as conclusion and the parameter and body
    wires as premises. A token reaching a premise **pushes** its bit -- ``0``
    for argument/parameter, ``1`` for result/body -- and leaves by the
    conclusion; a token reaching the conclusion **pops** the top bit and
    leaves by that premise. Thus the bits pushed at an abstraction route the
    token through the application it meets, and vice versa. On any other box
    (a constant) the token simply turns around.

    Parameters:
        box : A box of a lambda diagram.
    """
    from discopy import closed
    n_ports = len(box.dom) + len(box.cod)
    dom = cod = n_ports * (object, )
    if not isinstance(box, (closed.Eval, closed.Coeval)):
        return Function(lambda stack, tag=0: (stack, tag), dom, cod)
    if isinstance(box, closed.Eval):
        # dom = (function, argument), cod = (result, )
        conclusion = 0 if box.left else 1
        premises = {0: 1 if box.left else 0, 1: 2}
    else:
        # dom = (body, ), cod = (context, parameter)
        conclusion = 1 if box.left else 2
        premises = {0: 2 if box.left else 1, 1: 0}
    premise_bit = {tag: bit for bit, tag in premises.items()}

    def inside(stack, tag):
        if tag != conclusion:                       # a premise wire: push
            return stack + (premise_bit[tag], ), conclusion
        if not stack:
            raise ValueError(
                f"Token cannot enter {box.name} with an empty stack.")
        *rest, bit = stack                          # the conclusion wire: pop
        return tuple(rest), premises[bit]
    return Function(inside, dom, cod)


def to_hypergraph(term) -> Hypergraph:
    """
    Compile a linear lambda term into a token-passing additive hypergraph.

    Each box becomes its :func:`token_function` and the wiring is read off the
    term's rooted combinatorial map (:meth:`discopy.closed.Term.to_map`):
    every undirected wire becomes the two opposite directed wires a token can
    travel, so each box both reads (on the way in) and writes (on the way out)
    all of its wires, and every directed wire has a unique consumer -- the
    hypergraph is right-monogamous. The map is annotated on the result as
    ``hypergraph.cmap`` so that :func:`to_term` can walk it back to a term.

    Parameters:
        term : A :class:`discopy.closed.Term`.
    """
    cmap = term.to_map()
    edge = list(cmap.edge)

    # Each undirected edge {p, q} gives two directed wires p -> q and q -> p.
    directed = {}
    for p, q in enumerate(edge):
        if p < q:
            directed[p, q], directed[q, p] = len(directed), len(directed) + 1
    out_wire = lambda p: directed[p, edge[p]]   # token leaving port p
    in_wire = lambda p: directed[edge[p], p]    # token arriving at port p

    box_wires = tuple(
        (tuple(map(in_wire, indices)),      # read on the way in
         tuple(map(out_wire, indices)))     # written on the way out
        for indices in cmap.box_port_indices)
    boundary = [*range(len(cmap.dom)),
                *range(cmap.n_ports - len(cmap.cod), cmap.n_ports)]

    hypergraph = Hypergraph(
        dom=len(boundary) * (object, ), cod=len(boundary) * (object, ),
        boxes=tuple(map(token_function, cmap.boxes)),
        wires=(tuple(map(out_wire, boundary)), box_wires,
               tuple(map(in_wire, boundary))),
        spider_types=len(directed) * (object, ))
    hypergraph.cmap = cmap
    return hypergraph


def to_term(hypergraph: Hypergraph, input_names=None):
    """
    Recover a linear lambda term from the map annotated on the hypergraph.

    A recursive walk from the root against the data-flow -- from each wire to
    the box producing it -- descending into the function then the argument at
    an application, and binding a fresh variable before descending into the
    body at an abstraction. This is a direct-style version of
    :meth:`discopy.cmap.CMap.to_term`, kept as an independent implementation
    to cross-check against; the result is alpha-equivalent to the term that
    produced the hypergraph.

    Parameters:
        hypergraph : A hypergraph built by :func:`to_hypergraph`.
        input_names : Optional names for the free-variable input wires.
    """
    from discopy import closed

    cmap = hypergraph.cmap
    cmap.assert_rooted_map()
    ports, edge = cmap.ports, cmap.edge
    boxes, box_port_indices = cmap.boxes, cmap.box_port_indices

    def term_type(obj):
        return obj if hasattr(obj, "inside") else obj.ob(obj)

    cod = term_type(cmap.cod)
    names = tuple(
        (f"x{i}" for i in range(len(cmap.dom)))
        if input_names is None else input_names)
    if len(names) != len(cmap.dom):
        raise ValueError(
            f"Expected {len(cmap.dom)} input names, got {len(names)}.")
    variables = tuple(
        cod.variable_factory(name, term_type(obj))
        for obj, name in zip(cmap.dom, names))
    counter = len(variables)

    def fresh(obj):
        nonlocal counter
        counter += 1
        return cod.variable_factory(f"x{counter - 1}", obj)

    def walk(port_idx, bound):
        """ The term read from the box producing the wire at ``port_idx``. """
        port_idx = edge[port_idx]
        if port_idx in bound:
            return bound[port_idx]
        port = ports[port_idx]
        if port.kind == PortKind.INPUT:
            return variables[port.i]
        box = boxes[port.depth]
        indices = box_port_indices[port.depth]
        dom_ports = [i for i in indices if ports[i].kind == PortKind.DOM]

        if isinstance(box, closed.Eval):
            func_port, arg_port = dom_ports if box.left else dom_ports[::-1]
            func, arg = walk(func_port, bound), walk(arg_port, bound)
            return cod.application_factory(func, arg, left=not box.left)

        if isinstance(box, closed.Coeval):
            body_port, = dom_ports
            parameter_port, = [
                i for i in indices
                if ports[i].kind == PortKind.COD and i != port_idx]
            variable = fresh(term_type(port.obj).exponent)
            body = walk(body_port, bound | {parameter_port: variable})
            return cod.abstraction_factory(variable, body, left=not box.left)

        if port.kind == PortKind.COD and not box.dom and len(box.cod) == 1:
            return cod.constant_factory(box.name, term_type(box.cod))
        raise ValueError(f"Unexpected port in term recovery: {port}")

    return walk(cmap.n_ports - 1, {})


def roundtrip(term):
    """
    Send a term to a token-passing additive hypergraph and back.

    >>> from discopy.closed import Ty
    >>> X = Ty("X")
    >>> assert roundtrip(X(lambda x: x)) == X(lambda x: x)
    """
    return to_term(to_hypergraph(term))
