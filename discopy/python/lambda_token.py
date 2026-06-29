# -*- coding: utf-8 -*-

"""
Token passing for linear lambda terms.

A linear lambda term (:class:`discopy.closed.Term`) compiles to a string
diagram whose boxes are evaluations (applications, ``Eval``) and coevaluations
(abstractions, ``Coeval``). This module gives that diagram a *token-passing*
semantics in the additive category of Python functions
(:mod:`discopy.python.additive`), in the spirit of the Geometry of Interaction:
a token walks the diagram carrying a **stack of bits**, and each
application/abstraction is a :class:`Function` that **pushes a bit when the
token reaches it from an output and pops one when it comes from an input**.

The machine is assembled as a :class:`discopy.python.additive.Hypergraph`,
which is *right-monogamous* (each wire has a unique consumer) and need not be
causal -- the shape of a token machine, where each undirected wire of the term
becomes a pair of opposite directed wires the token can travel along. Walking
the token from the root recovers the term, so we get a round trip

    term -> additive hypergraph -> term

up to alpha-equivalence (see :func:`roundtrip`).

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    apply_function
    curry_function
    to_hypergraph
    to_term
    roundtrip
"""

from __future__ import annotations

from discopy.cmap import PortKind
from discopy.python.additive import Function, Hypergraph


def _token_function(box, push_bit):
    """
    The token-passing :class:`Function` shared by applications and
    abstractions: the box has ``len(box.dom) + len(box.cod)`` wires meeting it,
    each both an entry (its domain) and an exit (its codomain). A token
    reaching the box on one of its *outputs* (a codomain wire) **pushes**
    ``push_bit`` and continues on the first domain wire; one reaching it on an
    *input* (a domain wire) **pops** the top bit. The bit distinguishes the
    two kinds of node so an application cancels the abstraction it meets.
    """
    n_dom, n_cod = len(box.dom), len(box.cod)
    n_ports = n_dom + n_cod

    def inside(stack, tag):
        if tag >= n_dom:               # an output (codomain) wire: push
            return stack + (push_bit, ), 0
        *rest, _ = stack               # an input (domain) wire: pop
        return tuple(rest), n_dom
    return Function(inside, (object, ) * n_ports, (object, ) * n_ports)


def apply_function(box) -> Function:
    """
    The token-passing :class:`Function` of an application (``Eval``) box, which
    pushes the bit ``0`` when entered from its output (the result wire).

    Parameters:
        box : The ``Eval`` box.
    """
    return _token_function(box, push_bit=0)


def curry_function(box) -> Function:
    """
    The token-passing :class:`Function` of an abstraction (``Coeval``) box,
    which pushes the bit ``1`` when entered from one of its outputs (the
    function and bound-parameter wires).

    Parameters:
        box : The ``Coeval`` box.
    """
    return _token_function(box, push_bit=1)


def _box_function(box):
    """ The token :class:`Function` for any box of a lambda diagram. """
    from discopy import closed
    if isinstance(box, closed.Eval):
        return apply_function(box)
    if isinstance(box, closed.Coeval):
        return curry_function(box)
    # A constant or variable box: the token simply turns around.
    n_ports = len(box.dom) + len(box.cod)
    return Function(
        lambda stack, tag=0: (stack, tag), (object, ) * n_ports,
        (object, ) * n_ports)


def to_hypergraph(term):
    """
    Compile a linear lambda term into a token-passing additive hypergraph.

    Each application becomes an :func:`apply_function` and each abstraction a
    :func:`curry_function`. The wiring comes from the term's rooted
    combinatorial map (:meth:`discopy.closed.Term.to_map`): every undirected
    wire becomes the two opposite directed wires a token can travel, so each
    box both reads (on the way in) and writes (on the way out) all of its
    wires, and every directed wire has a unique consumer -- the hypergraph is
    right-monogamous. The term's lambda structure is annotated on the result
    (as ``hypergraph.lambda_meta``) so :func:`to_term` walks it back to a term.

    Parameters:
        term : A :class:`discopy.closed.Term`.
    """
    cmap = term.to_map()
    edge = list(cmap.edge)

    # Each undirected edge {p, q} gives two directed wires p->q and q->p.
    directed, n_spiders = {}, 0
    for p, q in enumerate(edge):
        if (p, q) not in directed:
            directed[(p, q)] = n_spiders
            directed[(q, p)] = n_spiders + 1
            n_spiders += 2
    out_wire = lambda p: directed[(p, edge[p])]   # token leaving port p
    in_wire = lambda p: directed[(edge[p], p)]    # token arriving at port p

    boxes, box_wires = [], []
    for box, indices in zip(cmap.boxes, cmap.box_port_indices):
        boxes.append(_box_function(box))
        box_wires.append((
            tuple(in_wire(p) for p in indices),     # read on the way in
            tuple(out_wire(p) for p in indices)))   # written on the way out

    boundary = list(range(len(cmap.dom))) + list(
        range(cmap.n_ports - len(cmap.cod), cmap.n_ports))
    dom_wires = tuple(out_wire(p) for p in boundary)
    cod_wires = tuple(in_wire(p) for p in boundary)

    hypergraph = Hypergraph(
        dom=(object, ) * len(boundary), cod=(object, ) * len(boundary),
        boxes=tuple(boxes), wires=(dom_wires, box_wires, cod_wires),
        spider_types=(object, ) * n_spiders)
    hypergraph.lambda_meta = {"cmap": cmap}
    return hypergraph


def to_term(hypergraph, input_names=None):
    """
    Recover a linear lambda term by walking the token machine.

    Starting on the root wire with an empty stack, a token walks the hypergraph
    against the data-flow -- from each wire to the box producing it. At an
    application it pushes a bit (output side) and descends into the function,
    then pops it (input side) and descends into the argument; at an abstraction
    it binds a fresh variable, pushes, descends into the body and pops. The
    visited boxes are assembled into a term, alpha-equivalent to the one that
    produced the hypergraph.

    Parameters:
        hypergraph : A hypergraph built by :func:`to_hypergraph`.
        input_names : Optional names for the free-variable input wires.
    """
    from discopy import closed

    cmap = hypergraph.lambda_meta["cmap"]
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
        raise ValueError
    variables = tuple(
        cod.variable_factory(name, term_type(obj))
        for obj, name in zip(cmap.dom, names))
    counter = len(variables)

    def fresh(obj):
        nonlocal counter
        variable = cod.variable_factory(f"x{counter}", obj)
        counter += 1
        return variable

    def walk(port_idx, bound, stack):
        """ Walk the token from ``port_idx``, returning ``(term, stack)``. """
        port_idx = edge[port_idx]
        if port_idx in bound:
            return bound[port_idx], stack
        port = ports[port_idx]
        if port.kind == PortKind.INPUT:
            return variables[port.i], stack
        box = boxes[port.depth]
        indices = box_port_indices[port.depth]
        local = indices.index(port_idx)

        if isinstance(box, closed.Eval):
            dom_ports = [i for i in indices if ports[i].kind == PortKind.DOM]
            func_port, arg_port = dom_ports if box.left\
                else tuple(reversed(dom_ports))
            stack, _ = apply_function(box)(stack, local)        # push (output)
            func, stack = walk(func_port, bound, stack)
            stack, _ = apply_function(box)(                     # pop (input)
                stack, indices.index(func_port))
            arg, stack = walk(arg_port, bound, stack)
            return cod.application_factory(func, arg, left=not box.left), stack

        if isinstance(box, closed.Coeval):
            exp = term_type(ports[port_idx].obj)
            body_port, = [i for i in indices if ports[i].kind == PortKind.DOM]
            parameter_port, = [
                i for i in indices
                if ports[i].kind == PortKind.COD and i != port_idx]
            stack, _ = curry_function(box)(stack, local)        # push (output)
            variable = fresh(exp.exponent)
            body, stack = walk(
                body_port, bound | {parameter_port: variable}, stack)
            stack, _ = curry_function(box)(                      # pop (input)
                stack, indices.index(body_port))
            return cod.abstraction_factory(
                variable, body, left=not box.left), stack

        if port.kind == PortKind.COD and not box.dom and len(box.cod) == 1:
            return cod.constant_factory(box.name, term_type(box.cod)), stack
        raise ValueError

    term, stack = walk(cmap.n_ports - 1, {}, ())
    if stack:
        raise RuntimeError(f"Token stack not empty: {stack}")
    return term


def roundtrip(term):
    """
    Send a term to a token-passing additive hypergraph and back.

    >>> from discopy.closed import Ty
    >>> X = Ty("X")
    >>> assert roundtrip(X(lambda x: x)) == X(lambda x: x)
    """
    return to_term(to_hypergraph(term))
