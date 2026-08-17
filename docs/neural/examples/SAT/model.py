# -*- coding: utf-8 -*-

"""
The SAT model: a diagram per formula, an interpretation, a solver.

Everything generic -- the compilation, the message passing, the cells, the
execution strategies -- is :mod:`discopy.neural`'s.  What is SAT's is in
this file and nowhere else:

1. the **roles** a wire can carry (:data:`MSG`, :data:`STATE`,
   :data:`CSTATE`), atomic types naming *what a wire carries* rather than
   how wide it is;
2. the **signatures** of the three generators, i.e. how many ports each has
   and which of them are one orbit under a permutation -- the members of a
   clause are, since a disjunction does not care which of its literals is
   which;
3. the **combinatorics** of a formula, :func:`incidence`, which draws the
   diagram: NeuroSAT-style literal splitting, where each variable becomes
   two literal nodes and the polarity lives in the graph rather than on the
   wires, because a wiring carries no per-edge sign;
4. a **readout** and a **loss**, the two ends of an unsupervised
   fill-in-the-assignment task.  There is no encoder: the formula *is* the
   diagram, so the initial state is a learned constant plus noise and
   nothing is injected.

The diagram
-----------

A CNF formula is a bipartite factor graph, drawn by
:func:`~discopy.neural.from_incidence`:

* one ``lit`` box per literal node, with one message port per relation it
  belongs to and a traced state loop;
* one ``clause`` box per clause, with one port per member;
* one ``flip`` box per variable, wiring its two literal nodes together --
  this is the only place polarity enters, and it is a *generator*, not a
  sign on an edge.

Two interpretations of that one wiring, flag-switched by
:data:`~config.Widths` and the ``stateful`` argument of :func:`build`:

* **stateless clause** -- ``CSTATE`` goes to ``Dim(0)``, which erases the
  clause loop and its ports, and the clause box carries a Deep-Sets
  :class:`~discopy.neural.Relation`;
* **stateful clause** -- ``CSTATE`` goes to ``Dim(state_dim)`` and the
  clause box carries a recurrent :class:`~discopy.neural.Site`, the
  NeuroSAT-faithful variant.

One diagram, two models, exactly as models A and C of ``examples/sudoku``
share a grid.

Note
----
The order in which the modules below are constructed is load-bearing: every
constructor draws from the global generator, so permuting them permutes
every weight in the model under a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from discopy.frobenius import Ty
from discopy.neural import (
    Dim, Iterate, MapNN, Mode, Orbit, Relation, Signature, Site, Sym,
    from_incidence)
from discopy.neural.core import box_ports

from config import CONSISTENCY, EPS, INIT_NOISE, K, WIDTHS, Widths

# --- 1. the roles ----------------------------------------------------------

#: A literal-to-relation wire: the message a literal node exchanges with a
#: clause it occurs in, or with the flip relation of its variable.
MSG = Ty("msg")

#: A literal node's recurrent state, on a traced loop.
STATE = Ty("state")

#: A clause's recurrent state, on a traced loop.  Sending it to ``Dim(0)``
#: erases the loop, which is how one diagram serves the stateful clause and
#: the stateless one.
CSTATE = Ty("cstate")


# --- 2. the signatures -----------------------------------------------------

def literal(degree: int = K) -> Signature:
    """
    A literal node: one message port per relation it belongs to, plus a
    state loop.

    The declared arity is only a default:
    :func:`~discopy.neural.from_incidence` resizes it per node, and one
    shared :class:`~discopy.neural.Site` fills every site whatever its
    degree.

    Parameters:
        degree : The number of relations the literal belongs to.

    Example
    -------
    >>> print(literal(2).cod)
    msg @ msg @ state @ state
    >>> literal(2).loops()
    ((2, 3),)
    """
    return Signature((Orbit(MSG, degree, Sym.PERM), Orbit(STATE, traced=True)))


def clause(size: int = K) -> Signature:
    """
    A clause: one port per member, permutation-symmetric, plus a state loop
    which an interpretation may erase.

    Parameters:
        size : The number of literals in the clause.

    Example
    -------
    >>> print(clause(3).cod)
    msg @ msg @ msg @ cstate @ cstate
    """
    return Signature((Orbit(MSG, size, Sym.PERM), Orbit(CSTATE, traced=True)))


def members(size: int = K) -> Signature:
    """
    The clause signature as a stateless module reads it, i.e. with the
    erased loop dropped.

    A :class:`~discopy.neural.Relation` has a single orbit by definition, so
    the stateless interpretation of a clause box needs this view of the same
    signature: the diagram still declares the loop, and ``Dim(0)`` erases
    its ports before any module sees them.

    Parameters:
        size : The number of members.
    """
    return Signature((Orbit(MSG, size, Sym.PERM), ))


def flip() -> Signature:
    """
    The flip relation of a variable: the two literal nodes of one variable,
    which is where polarity lives.

    Example
    -------
    >>> print(flip().cod)
    msg @ msg
    """
    return members(2)


def factor_ob(widths: Widths, clause_state: int = 0) -> dict:
    """ The ``Dim`` each role carries; ``clause_state=0`` erases the loop. """
    return {MSG: Dim(widths.dim), STATE: Dim(widths.state_dim),
            CSTATE: Dim(clause_state)}


def widths_of(ob) -> dict:
    """ The same mapping with plain integer widths, as a cell reads it. """
    return {role: sum(dim.inside) for role, dim in ob.items()}


# --- 3. the combinatorics, and the diagram ---------------------------------

def literal_of(dimacs: int) -> int:
    """
    The literal-node index of a DIMACS literal: ``2 * v`` for the positive
    literal of variable ``v``, ``2 * v + 1`` for its negation.

    Example
    -------
    >>> [literal_of(x) for x in (1, -1, 3, -3)]
    [0, 1, 4, 5]
    """
    return 2 * (dimacs - 1) if dimacs > 0 else 2 * (-dimacs - 1) + 1


def incidence(instances) -> tuple:
    """
    The bipartite incidence of a batch of formulas: per literal node, the
    relations it belongs to -- the clauses it occurs in, then the flip
    relation of its variable -- together with the name of each relation.

    Several instances are laid out side by side, which *is* their monoidal
    product: the disjoint union of their factor graphs, built in one pass
    rather than by folding ``@`` over the members.

    Parameters:
        instances : The formulas, as :class:`~dataset.Instance` objects.

    Returns:
        The incidence lists and the per-relation names.

    Example
    -------
    >>> from dataset import Instance
    >>> one = Instance(3, ((1, -2, 3), ), 0.33)
    >>> lists, names = incidence((one, ))
    >>> lists
    ((0, 1), (1,), (2,), (0, 2), (0, 3), (3,))
    >>> names
    ('clause', 'flip', 'flip', 'flip')
    """
    lists: list = []
    names: list = []
    relations = 0
    for instance in instances:
        local: list = [[] for _ in range(instance.n_literals)]
        for index, formula in enumerate(instance.clauses):
            for dimacs in formula:
                local[literal_of(dimacs)].append(relations + index)
        for variable in range(instance.n):
            for parity in (0, 1):
                local[2 * variable + parity].append(
                    relations + instance.m + variable)
        lists += [tuple(entry) for entry in local]
        names += ["clause"] * instance.m + ["flip"] * instance.n
        relations += instance.m + instance.n
    return tuple(lists), tuple(names)


def factor_graph(instances, category=None):
    """
    The factor graph of a batch of formulas, as a closed map in
    ``discopy.frobenius``.

    Parameters:
        instances : The formulas.
        category : The source category, ``discopy.frobenius`` by default.

    Example
    -------
    >>> from dataset import Instance
    >>> one = Instance(3, ((1, 2, 3), (-1, -2, 3)), 0.67)
    >>> graph = factor_graph((one, ))
    >>> len(graph.boxes), graph.n_ports // 2
    (11, 20)
    >>> sorted({box.name for box in graph.boxes})
    ['clause', 'flip', 'lit']
    """
    lists, names = incidence(instances)
    kwargs = {} if category is None else {"category": category}
    return from_incidence(
        lists, literal(), {"clause": clause(), "flip": flip()},
        node_name="lit", relation_name=names, **kwargs)


@lru_cache(maxsize=4096)
def interned(instance):
    """
    The factor graph of one formula, interned, so that evaluating the same
    instance twice compiles it once.

    Parameters:
        instance : The formula.
    """
    return factor_graph((instance, ))


# --- the index tensors a loss and a decode read ----------------------------

@dataclass
class Batch:
    """
    A batch of formulas run as one diagram, with the index tensors its loss
    and its decode read.

    Not :class:`discopy.neural.Batch`: that folds ``@`` over the members,
    which costs a quadratic number of port relabelings, whereas
    :func:`incidence` lays the same disjoint union out in one pass.  The
    result is the same map up to the order of its boxes -- see
    ``NOTES.md``.

    Every index is a *global* one, i.e. into the concatenation over the
    batch, so that a whole batch is one gather and one scatter:

    Parameters:
        instances : The formulas.
        diagram : Their factor graph.
        positive : Per variable, the literal-node index of its positive
                   literal; the negative one is the next index.
        clause_var : Per clause and member, the global variable index.
        clause_positive : Whether that member is a positive literal.
        clause_owner : Per clause, the instance it belongs to.
        clause_count : Per instance, its number of clauses.

    Example
    -------
    >>> from dataset import Instance
    >>> pair = (Instance(2, ((1, -2), ), 0.5), Instance(2, ((-1, 2), ), 0.5))
    >>> batch = Batch.of(pair)
    >>> batch.positive.tolist(), batch.clause_owner.tolist()
    ([0, 2, 4, 6], [0, 1])
    >>> batch.clause_var.tolist(), batch.clause_positive.tolist()
    ([[0, 1], [2, 3]], [[True, False], [False, True]])
    """

    instances: tuple
    diagram: object
    positive: torch.Tensor
    clause_var: torch.Tensor
    clause_positive: torch.Tensor
    clause_owner: torch.Tensor
    clause_count: torch.Tensor

    @classmethod
    def of(cls, instances, diagram=None) -> Batch:
        """
        The batch of a tuple of instances, building its diagram unless one
        is given.

        Parameters:
            instances : The formulas.
            diagram : Their factor graph, built here by default.
        """
        instances = tuple(instances)
        positive, clause_var, clause_positive, owner = [], [], [], []
        literals, variables = 0, 0
        for index, instance in enumerate(instances):
            positive.append(literals + 2 * np.arange(instance.n))
            signed = np.asarray(instance.clauses, dtype=np.int64)
            clause_var.append(variables + np.abs(signed) - 1)
            clause_positive.append(signed > 0)
            owner.append(np.full(instance.m, index))
            literals += instance.n_literals
            variables += instance.n

        def as_long(arrays):
            return torch.as_tensor(np.concatenate(arrays), dtype=torch.long)

        return cls(
            instances, factor_graph(instances) if diagram is None else diagram,
            as_long(positive), as_long(clause_var),
            torch.as_tensor(np.concatenate(clause_positive)),
            as_long(owner),
            torch.as_tensor([instance.m for instance in instances],
                            dtype=torch.long))

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def n_variables(self) -> int:
        """ The variables of the whole batch. """
        return len(self.positive)

    def to(self, device) -> Batch:
        """ The same batch with its index tensors on a device. """
        return Batch(
            self.instances, self.diagram, self.positive.to(device),
            self.clause_var.to(device), self.clause_positive.to(device),
            self.clause_owner.to(device), self.clause_count.to(device))


def check_layout(model, batch) -> None:
    """
    Assert that the ``i``-th site of the literal family is the ``i``-th
    literal node of the batch.

    The readout, the loss and the decode all index the literal axis by
    global literal-node index, and that they may is a property of how
    :func:`~discopy.neural.from_incidence` orders its boxes -- node boxes
    first, in incidence order -- rather than something declared anywhere.
    So it is checked against the wiring instead of assumed.

    Parameters:
        model : The model whose interpretation compiles the diagram.
        batch : The batch to check.
    """
    interaction = model.map.compile(batch.diagram)
    heads = interaction.heads[model.answer]
    cmap = interaction.cmap
    expected = sum(instance.n_literals for instance in batch.instances)
    assert len(heads) == expected, \
        f"{len(heads)} literal sites for {expected} literal nodes"
    for index, port in enumerate(heads):
        assert cmap.boxes[index].name == "lit", \
            f"box {index} is a {cmap.boxes[index].name}, not a literal"
        assert port in box_ports(cmap, index), \
            f"the {index}-th literal state is not on the {index}-th box"


# --- 4. the two ends of the task -------------------------------------------

class Readout(torch.nn.Linear):
    """
    One logit per literal node, read off its state.

    A variable's probability is that of its *positive* literal; the
    negative literal has a logit of its own, and that the two disagree is a
    measurable quantity rather than a structural impossibility, which is
    what :func:`consistency` reports and penalises.

    Example
    -------
    >>> Readout(4)(torch.zeros(2, 6, 4)).shape
    torch.Size([2, 6, 1])
    """
    def __init__(self, state_dim: int):
        super().__init__(state_dim, 1)


def variable_logits(literal_logits, batch: Batch):
    """
    The per-variable logit of a batch, i.e. the logit of its positive
    literal.

    Parameters:
        literal_logits : One logit per literal node, ``(rows, literals)``.
        batch : The batch the logits belong to.
    """
    return literal_logits.index_select(1, batch.positive)


def consistency(literal_logits, batch: Batch):
    """
    How far the two literals of a variable are from being complementary,
    as a mean squared probability gap.

    The wiring says a variable's two literal nodes are related -- they are
    the two members of a ``flip`` box -- but nothing makes the learned cell
    give them opposite beliefs.  This is the residual of that, used as a
    small auxiliary loss and reported as a diagnostic.

    Parameters:
        literal_logits : One logit per literal node.
        batch : The batch the logits belong to.
    """
    positive = literal_logits.index_select(1, batch.positive)
    negative = literal_logits.index_select(1, batch.positive + 1)
    return (positive.sigmoid() + negative.sigmoid() - 1).square().mean()


def clause_scores(literal_logits, batch: Batch):
    """
    The smooth clause scores :math:`1 - \\prod_{l \\in c} (1 - p_l)`, in
    logarithm.

    Computed in log space throughout: a literal's falsity log-probability
    is ``logsigmoid(-s)`` for ``s`` its signed logit, the product is their
    sum, and the score follows by ``log(-expm1(.))``, which is exact where
    ``1 - exp`` would cancel.

    Parameters:
        literal_logits : One logit per literal node.
        batch : The batch the logits belong to.

    Returns:
        ``log(score)`` per clause, of shape ``(rows, clauses)``.
    """
    signed = variable_logits(literal_logits, batch).index_select(
        1, batch.clause_var.reshape(-1)).reshape(
            len(literal_logits), *batch.clause_var.shape)
    signed = torch.where(batch.clause_positive, signed, -signed)
    false = torch.nn.functional.logsigmoid(-signed).sum(-1)
    return torch.log(-torch.expm1(false.clamp(max=-1e-7)) + EPS)


def smooth_sat_loss(literal_logits, batch: Batch,
                    consistency_weight: float = CONSISTENCY):
    """
    The unsupervised loss: :math:`-\\sum_c \\log(\\text{score}_c)`, averaged
    over the clauses of an instance and then over the instances, plus the
    literal-consistency term.

    No assignment is supervised.  A satisfiable formula has many solutions
    and supervising one of them is ill-posed; what the loss asks for is
    that *every clause* be satisfied, which is the actual specification.

    Parameters:
        literal_logits : One logit per literal node.
        batch : The batch the logits belong to.
        consistency_weight : The weight of the auxiliary term.
    """
    per_clause = -clause_scores(literal_logits, batch)
    totals = torch.zeros(
        len(literal_logits), len(batch), dtype=per_clause.dtype,
        device=per_clause.device).index_add(1, batch.clause_owner, per_clause)
    loss = (totals / batch.clause_count).mean()
    if not consistency_weight:
        return loss
    return loss + consistency_weight * consistency(literal_logits, batch)


@torch.no_grad()
def decode(literal_logits, batch: Batch):
    """
    Round the assignment at ``0.5`` and check it exactly.

    The verifier is free and exact -- count the unsatisfied clauses -- which
    is the one place this domain is kinder than sudoku: no learned critic
    is needed to know whether an answer is right.

    Parameters:
        literal_logits : One logit per literal node.
        batch : The batch the logits belong to.

    Returns:
        The assignment ``(rows, variables)``, the number of unsatisfied
        clauses per instance and whether each instance is solved.
    """
    assignment = variable_logits(literal_logits, batch) > 0
    values = assignment.index_select(
        1, batch.clause_var.reshape(-1)).reshape(
            len(literal_logits), *batch.clause_var.shape)
    unsatisfied = (values != batch.clause_positive).all(-1)
    counts = torch.zeros(
        len(literal_logits), len(batch), dtype=torch.long,
        device=assignment.device).index_add(
            1, batch.clause_owner, unsatisfied.long())
    return assignment, counts, counts == 0


# --- the model -------------------------------------------------------------

class Model(torch.nn.Module):
    """
    A learned SAT solver: a :class:`~discopy.neural.MapNN`, a learned
    initial belief and a readout.

    There is no encoder.  The formula is the diagram, so the only thing
    written into the initial state is one learned vector shared by every
    literal node, plus a small Gaussian perturbation without which every
    literal of a symmetric formula would carry the same belief for ever.
    Nothing is re-injected: the solver runs with ``inject=False`` and the
    state carries the whole run.

    Parameters:
        interpretation : The :class:`~discopy.neural.MapNN`.
        readout : The per-literal logit head.
        initial : The learned initial literal state.
        clause_initial : The learned initial clause state, or ``None`` when
                         the clause loop is erased.
        noise : The standard deviation of the initial perturbation.
    """
    def __init__(self, interpretation: MapNN, readout: Readout,
                 initial: torch.nn.Parameter,
                 clause_initial: torch.nn.Parameter = None,
                 noise: float = INIT_NOISE):
        super().__init__()
        self.map = interpretation
        self.readout = readout
        self.h0 = initial
        self.c0 = clause_initial
        self.noise = noise
        self.answer = ("lit", STATE)

    @property
    def rounds(self) -> int:
        """ The message-passing rounds the solver runs by default. """
        return self.map.solver.rounds

    def sites(self, batch: Batch) -> int:
        """ The literal nodes of a batch, read off its compiled diagram. """
        return self.map.sites(batch.diagram, self.answer)

    def initial(self, batch: Batch, rows: int = 1, generator=None):
        """
        The initial flat state: the learned belief on every literal's state
        loop, plus noise.

        Parameters:
            batch : The batch to run.
            rows : The number of independent runs of the same batch, which
                   is how a rollout study draws several samples of the
                   noise.
            generator : The ``torch.Generator`` behind the perturbation.
        """
        sites = self.sites(batch)
        values = {self.answer: self._spread(self.h0, rows, sites, generator)}
        if self.c0 is not None:
            clauses = ("clause", CSTATE)
            values[clauses] = self._spread(
                self.c0, rows, self.map.sites(batch.diagram, clauses),
                generator)
        return self.map.initial(batch.diagram, values, rows=rows)

    def _spread(self, vector, rows: int, sites: int, generator):
        """ One learned vector on every site, perturbed. """
        spread = vector.expand(rows, sites, len(vector))
        if not self.noise:
            return spread.contiguous()
        return spread + self.noise * torch.randn(
            spread.shape, generator=generator, device=vector.device,
            dtype=vector.dtype)

    def logits(self, batch: Batch, state):
        """ One logit per literal node, of shape ``(rows, literals)``. """
        return self.readout(
            self.map.read(batch.diagram, state, self.answer)).squeeze(-1)

    def forward(self, batch: Batch, state=None, deep: bool = False,
                rows: int = 1, generator=None, **overrides):
        """
        The literal logits of a whole run, at every round when ``deep`` and
        at the last one otherwise.

        Parameters:
            batch : The batch to run.
            state : The flat initial messages, built here by default.
            deep : Whether to return every round.
            rows : The independent runs, when the state is built here.
            generator : The generator behind the initial noise.
            overrides : Test-time compute, i.e. ``rounds``.
        """
        state = self.initial(batch, rows, generator) if state is None \
            else state
        states = self.map(batch.diagram, state, deep=deep, **overrides)
        return [self.logits(batch, one) for one in states] if deep \
            else self.logits(batch, states)


def build(widths: Widths = None, rounds: int = 32, stateful: bool = False,
          noise: float = INIT_NOISE, cache: int = 128,
          pool: str = "mean") -> Model:
    """
    The factor-graph model of Part 1.

    Parameters:
        widths : The widths, ``WIDTHS["factor"]`` by default.
        rounds : The message-passing rounds trained at.
        stateful : Whether the clause box is a recurrent
                   :class:`~discopy.neural.Site` rather than a stateless
                   :class:`~discopy.neural.Relation`, i.e. whether
                   ``CSTATE`` is erased.
        noise : The standard deviation of the initial perturbation.
        cache : How many compiled diagrams the interpretation keeps; a
                whole pool of batches should fit, or every epoch pays for
                its compilation again.
        pool : How a literal node pools its incoming messages.  Literal
               degrees vary a lot in a random formula, so the default is a
               mean, which keeps a cell's input scale independent of its
               degree.

    Example
    -------
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch...>
    >>> model = build(Widths(dim=8, state_dim=16, hidden=16), rounds=4)
    >>> count_parameters(model) > 0
    True
    """
    widths = widths or WIDTHS["factor"]
    ob = factor_ob(widths, widths.state_dim if stateful else 0)
    sizes = widths_of(ob)
    site = Site(literal(), sizes, {STATE: Mode.STATE}, hidden=widths.hidden,
                depth=2, pool=pool, recurrent="gru", emit=True)
    unit = Site(clause(), sizes, {CSTATE: Mode.STATE}, hidden=widths.hidden,
                depth=2, pool="sum", recurrent="gru", emit=True) if stateful \
        else Relation(members(), {MSG: widths.dim}, hidden=widths.hidden,
                      pool="sum")
    negation = Relation(flip(), {MSG: widths.dim}, hidden=widths.hidden,
                        pool="sum")
    readout = Readout(widths.state_dim)
    initial = torch.nn.Parameter(torch.zeros(widths.state_dim))
    clause_initial = torch.nn.Parameter(torch.zeros(widths.state_dim)) \
        if stateful else None
    return Model(
        MapNN(ob, {"lit": site, "clause": unit, "flip": negation},
              solver=Iterate(rounds=rounds, inject=False), cache=cache),
        readout, initial, clause_initial, noise=noise)


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_checkpoint(module, path, strict: bool = True):
    """
    Load a checkpoint into a model.

    Parameters:
        module : The model to load into.
        path : A bare ``state_dict`` or a ``{"state_dict": ..., ...}``
               record.
        strict : Whether to require every key to match, as it should.

    Returns:
        Whatever ``torch.load`` read, so that the caller can still see the
        history and metadata beside the weights.
    """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    weights = stored["state_dict"] if isinstance(stored, dict) \
        and "state_dict" in stored else stored
    module.load_state_dict(weights, strict=strict)
    return stored
