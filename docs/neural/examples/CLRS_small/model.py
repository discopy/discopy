# -*- coding: utf-8 -*-

"""
The CLRS model: a diagram per batch of graphs, an interpretation, a solver.

Everything generic -- the compilation, the message passing, the cells, the
execution strategies -- is :mod:`discopy.neural`'s.  What is CLRS's is in
this file and nowhere else:

1. the **roles** a wire can carry (:data:`MSG`, :data:`STATE`, ...), atomic
   types naming *what a wire carries* rather than how wide it is;
2. the **signatures** of the three generators -- a node, an edge and the
   graph-level readout relation -- i.e. how many ports each has and which
   of them are one orbit under a permutation, and the one cell that is not
   a stock one, :class:`Link`;
3. the **combinatorics** of a sample, :func:`incidence`, which draws the
   diagram out of an adjacency matrix -- or out of the size alone, for the
   two algorithms that answer about every pair;
4. the **encoders, decoders and losses** of the benchmark's feature types,
   which are the two ends of an algorithmic-reasoning task.

The diagram
-----------

A CLRS sample over ``n`` nodes is a bipartite incidence graph, drawn by
:func:`~discopy.neural.from_incidence`:

* one ``node`` box per node, with one message port per relation it belongs
  to, a traced recurrent state and a traced loop carrying its encoded
  inputs;
* one ``edge`` box per edge -- or per *pair*, on the complete-graph diagram
  of :data:`~dataset.DENSE` -- with one port per endpoint, a traced state
  that :data:`Dim(0)` may erase, and a traced *carried* block of per-edge
  inputs, which a cell emits unchanged so that per-edge inputs need no
  library change;
* one ``readout`` box per sample, a relation containing **every** node under
  a name of its own.  It is where graph-level features live, it is what
  makes ``minimum`` -- which has no edges at all -- solvable, and it has the
  side effect that every node has degree at least one, so an isolated node
  of an Erdos-Renyi sample still reaches :func:`from_incidence`.

Two rounds to a hop
-------------------

A node speaks to a node through a box: ``node -> edge -> node`` and
``node -> readout -> node`` are both two rounds of message passing.  One
step of the algorithm being imitated is therefore :data:`HOPS` rounds, not
one -- the same factor of two the sudoku study measured as its hop law.
:func:`alignment` is where the two clocks are matched, once, and
:meth:`Model.checkpoints` and :meth:`Model.hint_targets` read it rather
than repeating it; :func:`rounds_of` is how long a run is, namely as long
as the trajectory it imitates.

Note
----
The order in which the modules below are constructed is load-bearing: every
constructor draws from the global generator, so permuting them permutes
every weight in the model under a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from discopy.frobenius import Ty
from discopy.neural import (
    Dim, FixedPoint, Iterate, MapNN, Mode, Orbit, Signature, Site, Sym, cells,
    from_incidence)
from discopy.neural.core import box_ports

from config import HINT_WEIGHT, POOL, WIDTHS, Widths, holding
from dataset import (
    CLASSES, DENSE, DIRECTED, MASKED, SPECS, Split, complete, edge_features,
    kind, probes)

# --- 1. the roles ----------------------------------------------------------

#: A node-to-relation wire: the message a node exchanges with an edge it is
#: an endpoint of, or with the readout relation of its sample.
MSG = Ty("msg")

#: A node's recurrent state, on a traced loop.
STATE = Ty("state")

#: A node's encoded inputs, on a traced loop the site reads but never
#: rewrites, so that a run carries its own inputs and can be stopped,
#: resumed and run deeper at test time without re-injection.
FEAT = Ty("feat")

#: An edge's recurrent state, on a traced loop.  Sending it to ``Dim(0)``
#: erases the edge machinery from the same diagram, which is Part 2's H1
#: ablation.
ESTATE = Ty("estate")

#: An edge's encoded weight, on a traced loop carried unchanged.
WEIGHT = Ty("weight")

#: The readout relation's recurrent state, i.e. the graph-level state.
GSTATE = Ty("gstate")

#: The rounds of message passing one step of the imitated algorithm costs:
#: a node reaches a node through a box, so a hop is two rounds.
HOPS = 2


class Grounded(FixedPoint):
    """
    The Jacobian-free fixed point, with the inputs kept attached.

    :class:`~discopy.neural.FixedPoint` under ``backward="last"``
    differentiates one round from ``state.detach()``.  That is the
    deep-equilibrium gradient and it is right for an interaction that
    *re-injects* its inputs, since then the final round reads them
    through ``inject`` and they stay in the graph.  Here they do not:
    the cells are resumable, so a run carries its own inputs on traced
    loops (:data:`FEAT`, :data:`WEIGHT`) and ``inject`` is ``False`` --
    and detaching the state therefore detaches the encoders too.
    Measured on ``bfs``: under ``FixedPoint(backward="last")`` all six
    encoder parameters have ``grad is None``, so an arm built on it
    would train frozen random encoders and its comparison with an
    ``Iterate`` arm would be two changes wide.

    The repair costs nothing, because those roles are *carried*: a site
    re-emits them unchanged, so the limit's values on them are bitwise
    the ones the encoders wrote.  Writing them back into the detached
    limit before the differentiated round therefore leaves every number
    of the forward pass alone and restores the one gradient path that
    was missing -- ``test_a_grounded_fixed_point_is_the_same_forward``
    and ``test_a_grounded_fixed_point_trains_its_encoders`` pin both
    halves.

    Parameters:
        rounds : The maximum rounds.
        tol : The residual to stop at, ``None`` never to test.
        inject : Whether every round re-adds the initial messages.
        carried : The ``(generator, role)`` families to write back.
    """
    def __init__(self, rounds: int = 32, tol: float = None,
                 inject: bool = False, carried=()):
        super().__init__(rounds, tol, inject, backward="last")
        self.carried = tuple(carried)

    def ground(self, interaction, limit, state):
        """ The detached limit, with the carried roles re-attached. """
        found = limit.detach()
        for key in self.carried:
            found = interaction.write(
                found, key, interaction.read(state, key))
        return found

    def run(self, interaction, state, deep: bool = False, rounds: int = None,
            tol: float = None):
        rounds = self.rounds if rounds is None else rounds
        tol = self.tol if tol is None else tol
        every, found = [], state
        with torch.no_grad():
            for _ in range(max(rounds - 1, 0)):
                step = interaction.advance(found, 1, self.inject)
                settled = tol is not None and bool(
                    (step - found).abs().amax() < tol)
                found = step
                if deep:
                    every.append(found)
                if settled:
                    break
        found = interaction.advance(
            self.ground(interaction, found, state), 1, self.inject)
        if deep:
            every.append(found)
        return found, every or [found]


#: The solvers a budget may name, by the two policies a
#: :class:`~discopy.neural.solver.Solver` carries: how many rounds to run,
#: and which of them are in the autograd graph.
#:
#: ``tol`` is ``None`` on all three, and that is Part 3's protocol rather
#: than a default: a residual stopping rule run *during training* shortens
#: the supervised sequence, so a fixed-point arm would differ from an
#: ``Iterate`` one in the gradient **and** in the effective depth, and a
#: two-change result measures neither.  So every arm trains at the
#: trajectory's rounds and the stopping rule is applied at evaluation, on
#: weights that are already fixed; see ``PART3.md``.
#:
#: The consequence is that ``backward`` is the whole of a fixed-point row:
#: with ``tol=None`` and ``inject=False``,
#: :class:`~discopy.neural.FixedPoint` under ``"full"`` is *bitwise*
#: :class:`~discopy.neural.Iterate` -- ``test_a_full_fixed_point_is_an_
#: iterate`` says so -- and only ``"last"`` is a row rather than a rename.
#: ``"fixedpoint"`` is the library's, kept so that the encoder measurement
#: :class:`Grounded` documents stays reproducible; ``"grounded"`` is the
#: one Part 3 trains.
SOLVERS = {
    "iterate": lambda backward, carried: Iterate(rounds=HOPS, inject=False),
    "fixedpoint": lambda backward, carried: FixedPoint(
        rounds=HOPS, tol=None, inject=False, backward=backward),
    "grounded": lambda backward, carried: Grounded(
        rounds=HOPS, tol=None, inject=False, carried=carried),
}


def alignment(rounds: int, hops: int = HOPS) -> tuple:
    """
    The correspondence between the two clocks, as one table: per
    checkpoint, the **round** it is read at and the **hint** it is
    supervised on.

    This is the most dangerous arithmetic in the study and it is therefore
    written once, here, and read from :meth:`Model.checkpoints` and
    :meth:`Model.hint_targets` rather than repeated in either.  Two facts
    it encodes:

    * a node reaches a node through a box, so one step of the imitated
      algorithm is ``hops`` rounds and the ``k``-th checkpoint is the state
      after round ``hops * (k + 1)`` -- supervising at a round that is not
      a multiple of ``hops`` would ask a factor graph to have propagated
      half a hop;
    * ``hints[0]`` is the trajectory's *initial condition* rather than
      something to predict, so the ``k``-th checkpoint is supervised on
      ``hints[k + 1]``.  This is ``clrs._src.evaluation.evaluate_hints``'
      own indexing: it scores hint prediction ``i`` against ``idx=i+1``.

    An off-by-one in either column is invisible in a loss curve -- the
    model half-learns the shifted target and every probe is uniformly
    mediocre -- so it is also pinned constructively, on a case whose
    answer can be computed by hand, by
    ``test_a_checkpoint_is_the_algorithm_step_it_says_it_is``.

    Parameters:
        rounds : The rounds a run performs.
        hops : The rounds one algorithm step costs.

    Returns:
        One ``(checkpoint, round, hint)`` triple per checkpoint, with
        ``round`` counted from one.

    Example
    -------
    >>> alignment(6)
    ((0, 2, 1), (1, 4, 2), (2, 6, 3))
    >>> alignment(7) == alignment(6), alignment(1)
    (True, ())
    """
    return tuple((step, hops * (step + 1), hint_of(step))
                 for step in range(rounds // hops))


def hint_of(checkpoint: int) -> int:
    """
    The hint a checkpoint is supervised on: ``hints[0]`` is the initial
    condition, so the ``k``-th checkpoint predicts ``hints[k + 1]``.

    Example
    -------
    >>> hint_of(0), hint_of(3)
    (1, 4)
    """
    return checkpoint + 1


def rounds_of(steps: int, hops: int = HOPS) -> int:
    """
    The rounds a trajectory of ``steps`` algorithm steps is run for: the
    **trajectory rule** of Part 2.

    Part 1 ran a constant 16 rounds for every sample of every algorithm,
    which is eight steps of an algorithm whose trajectory is three steps
    long on one row of the table and sixty-four on another.  Under this
    rule the depth is a property of the sample, so ``floyd_warshall`` at
    ``n = 64`` is run the 64 steps its ``k`` loop takes, and "the model
    did not finish" stops being a rival explanation for every negative
    result.

    Parameters:
        steps : The algorithm steps to cover.
        hops : The rounds one step costs.

    Example
    -------
    >>> rounds_of(8), rounds_of(64)
    (16, 128)
    >>> len(alignment(rounds_of(5))) == 5
    True
    """
    return hops * max(1, steps)


# --- 2. the signatures -----------------------------------------------------

def node(degree: int = 1) -> Signature:
    """
    A node: one message port per relation it belongs to, plus a state loop
    and an input loop.

    The declared arity is only a default:
    :func:`~discopy.neural.from_incidence` resizes it per node and one
    shared :class:`~discopy.neural.Site` fills every site whatever its
    degree.

    Parameters:
        degree : The number of relations the node belongs to.

    Example
    -------
    >>> print(node(2).cod)
    msg @ msg @ state @ state @ feat @ feat
    >>> node(2).loops()
    ((2, 3), (4, 5))
    """
    return Signature((Orbit(MSG, degree, Sym.PERM), Orbit(STATE, traced=True),
                      Orbit(FEAT, traced=True)))


def edge(directed: bool = False) -> Signature:
    """
    An edge: one port per endpoint, plus a state loop and a carried block
    of per-edge inputs.

    ``Sym.PERM`` on the endpoints is the honest signature of an undirected
    graph, and it is what CLRS's own samplers draw for seven of the eight
    algorithms -- ``check_equivariant`` is then owed on the cell, and is
    measured.  ``Sym.NONE`` is the honest signature of the eighth: a
    directed edge answers its source and its target differently, so its
    ports are *not* an orbit and no equation is owed.  Which of the two an
    algorithm gets is read off the sampler, in :data:`dataset.DIRECTED`.

    Example
    -------
    >>> print(edge().cod)
    msg @ msg @ estate @ estate @ weight @ weight
    >>> edge().orbits[0].sym, edge(directed=True).orbits[0].sym
    (<Sym.PERM: 'perm'>, <Sym.NONE: 'none'>)
    """
    return Signature((Orbit(MSG, 2, Sym.NONE if directed else Sym.PERM),
                      Orbit(ESTATE, traced=True), Orbit(WEIGHT, traced=True)))


def readout(size: int = 1) -> Signature:
    """
    The graph-level readout relation: one port per node of the sample,
    plus a state loop of its own.

    Parameters:
        size : The number of nodes.
    """
    return Signature((Orbit(MSG, size, Sym.PERM), Orbit(GSTATE, traced=True)))


class Link(cells.Cell):
    """
    An edge box: two message ports, a block of carried per-edge inputs, and
    a recurrent state **that may be absent**.

    The one custom cell of the study, and it exists because Part 2 turns
    two knobs on the edge and the stock :class:`~discopy.neural.Site` holds
    neither.

    *Direction.*  A ``Site`` pools its message orbit, so it cannot answer
    its source differently from its target; ``dag_shortest_paths`` is
    directed and needs it to.  Under ``Sym.NONE`` the two legs are read in
    node-index order -- which endpoint is the source is a bit on the carry,
    written by :attr:`~dataset.Split.edge_inputs` -- and one map answers
    both at once, so the arithmetic is not symmetric and the signature does
    not claim it is.

    *Memory.*  ``ESTATE`` sent to ``Dim(0)`` erases the state ports from
    the diagram, and a ``Site`` cannot be built without a state to carry
    (it raises).  That is H1's node-only arm: the same wiring, the same
    messages through the same boxes, and nothing that remembers a pair
    between rounds.  With no state the belief the cell emits from is the
    pooled encoding of the round itself, so the two arms differ in
    *persistence* and in nothing else.

    Parameters:
        signature : The signature of the edge, see :func:`edge`.
        widths : The width each atomic role carries; ``ESTATE`` may be zero.
        hidden : The width of the hidden layers.
        depth : The number of linear layers of the encoder.
        pool : The key in :data:`discopy.neural.cells.POOL` the two legs
               are pooled with, when they are an orbit.
        recurrent : The key in :data:`discopy.neural.cells.RECURRENT` of
                    the update cell.

    Note
    ----
    The submodules are built in the order ``encode``, ``update``, ``norm``,
    ``emit``, as a ``Site`` builds them, so that a cell drawn under a seed
    draws the same numbers in the same order.

    Example
    -------
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch...>
    >>> widths = {MSG: 4, ESTATE: 6, WEIGHT: 4}
    >>> cell = Link(edge(), widths, hidden=8)
    >>> cell(torch.zeros(2, 2 * 4 + 2 * 6 + 2 * 4)).shape
    torch.Size([2, 28])

    It keeps the equation its signature declares, up to the reordering of
    a float64 reduction:

    >>> from discopy.neural import check_equivariant
    >>> float(check_equivariant(
    ...     Link(edge(), widths, hidden=8).double(), edge(), widths)[MSG])
    0.0

    Erasing the state erases its ports, and the cell still runs:

    >>> stateless = {MSG: 4, ESTATE: 0, WEIGHT: 4}
    >>> Link(edge(), stateless, hidden=8)(torch.zeros(2, 16)).shape
    torch.Size([2, 16])
    """
    def __init__(self, signature: Signature, widths, hidden: int,
                 depth: int = 2, pool: str = "mean", recurrent: str = "gru"):
        super().__init__(signature, widths)
        self.mode = {ESTATE: Mode.STATE, WEIGHT: Mode.CARRY}
        self.directed = signature.orbits[0].sym == Sym.NONE
        self.pooling = cells.POOL[pool]
        self.states = self.roles(Mode.STATE)
        self.given = self.roles(Mode.CARRY)
        self.carried = sum(self.widths[role] for role in self.given)
        #: The width of the belief the cell emits from: its state when it
        #: has one, the pooled encoding of the round when it has not.
        self.belief = self.widths[self.states[0]] if self.states else hidden

        self.encode = _mlp(
            (2 if self.directed else 1) * self.leg + self.carried,
            hidden, depth)
        self.update = cells.RECURRENT[recurrent](hidden, self.belief) \
            if self.states else None
        self.norm = torch.nn.LayerNorm(self.belief)
        self.emit = torch.nn.Linear(
            self.belief + (2 if self.directed else 1) * self.leg,
            (2 if self.directed else 1) * self.leg)

    def forward(self, x):
        arity, places = self.places(x.shape[-1])
        if arity != 2:
            raise ValueError(f"an edge has two endpoints, not {arity}")
        message = x[:, places[self.orbit[0]]].reshape(-1, 2, self.leg)
        given = [x[:, places[role]] for role in self.given]

        if self.directed:
            pooled = self.encode(torch.cat(
                [message.reshape(-1, 2 * self.leg)] + given, -1))
        else:
            pooled = self.pooling(self.encode(torch.cat(
                [message] + [one.unsqueeze(1).expand(-1, 2, -1)
                             for one in given], -1)))
        belief = self.norm(pooled if self.update is None else self.update(
            pooled, x[:, places[self.states[0]]]))

        if self.directed:
            legs = self.emit(torch.cat(
                [belief, message.reshape(-1, 2 * self.leg)], -1))
        else:
            legs = self.emit(torch.cat([
                belief.unsqueeze(1).expand(-1, 2, -1), message], -1)).reshape(
                    -1, 2 * self.leg)
        emitted = dict(zip(self.states, [belief]))
        emitted.update(zip(self.given, given))
        out = torch.cat([legs] + [
            emitted[atom] for orbit in self.signature.orbits[1:]
            for _ in range(orbit.copies * orbit.arity)
            for atom in orbit.role if self.widths[atom]], -1)
        assert out.shape == x.shape, "the cell changed its port widths"
        return out


def _mlp(dom: int, hidden: int, depth: int) -> torch.nn.Sequential:
    """ ``depth`` linear layers from ``dom`` to ``hidden``, ReLU between. """
    layers: list = []
    for index in range(depth):
        if index:
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden if index else dom, hidden))
    return torch.nn.Sequential(*layers)


def graph_ob(widths: Widths, edge_dim: int = None) -> dict:
    """ The ``Dim`` each role carries; ``edge_dim=0`` erases the edges. """
    return {MSG: Dim(widths.dim), STATE: Dim(widths.state_dim),
            FEAT: Dim(widths.dim), GSTATE: Dim(widths.graph_dim),
            ESTATE: Dim(widths.edge_dim if edge_dim is None else edge_dim),
            WEIGHT: Dim(widths.dim)}


def widths_of(ob) -> dict:
    """ The same mapping with plain integer widths, as a cell reads it. """
    return {role: sum(dim.inside) for role, dim in ob.items()}


# --- 3. the combinatorics, and the diagram ---------------------------------

def incidence(edges, size: int) -> tuple:
    """
    The bipartite incidence of a batch of graphs: per node, the relations
    it belongs to -- the edges it is an endpoint of, then the readout
    relation of its sample -- together with the name of each relation.

    Several samples are laid out side by side, which *is* their monoidal
    product: the disjoint union of their incidence graphs, built in one
    pass rather than by folding ``@`` over the members.

    Parameters:
        edges : Per sample, an integer array of shape ``(m, 2)`` of
                undirected edges.
        size : The number of nodes of every sample of the batch.

    Returns:
        The incidence lists and the per-relation names.

    Example
    -------
    >>> pairs = [np.array([[0, 1]]), np.zeros((0, 2), dtype=int)]
    >>> lists, names = incidence(pairs, 2)
    >>> lists
    ((0, 1), (0, 1), (2,), (2,))
    >>> names
    ('edge', 'readout', 'readout')
    """
    lists: list = []
    names: list = []
    relations = 0
    for pairs in edges:
        local: list = [[] for _ in range(size)]
        for index, (one, other) in enumerate(pairs):
            local[int(one)].append(relations + index)
            local[int(other)].append(relations + index)
        for entry in local:
            entry.append(relations + len(pairs))
        lists += [tuple(entry) for entry in local]
        names += ["edge"] * len(pairs) + ["readout"]
        relations += len(pairs) + 1
    return tuple(lists), tuple(names)


def graph(edges, size: int, category=None):
    """
    The incidence graph of a batch of samples, as a closed map in
    ``discopy.frobenius``.

    Parameters:
        edges : Per sample, its undirected edge array.
        size : The number of nodes of every sample.
        category : The source category, ``discopy.frobenius`` by default.

    Example
    -------
    >>> drawn = graph([np.array([[0, 1], [1, 2]])], 3)
    >>> [box.name for box in drawn.boxes]
    ['node', 'node', 'node', 'edge', 'edge', 'readout']
    >>> drawn.n_ports // 2
    18
    """
    lists, names = incidence(edges, size)
    kwargs = {} if category is None else {"category": category}
    return from_incidence(
        lists, node(), {"edge": edge(), "readout": readout()},
        node_name="node", relation_name=names, **kwargs)


@lru_cache(maxsize=16)
def dense_graph(size: int, samples: int):
    """
    The incidence graph of a batch of :data:`~dataset.DENSE` samples: one
    edge box per *pair*, the same wiring for every sample of a size.

    Which is why it is cached, and why the two edge-state showcases are
    cheaper to run than their box count suggests: their diagram does not
    depend on the sample, so a whole split compiles **once** rather than
    once per batch, however superlinear compiling one is.

    Parameters:
        size : The number of nodes of every sample.
        samples : The number of samples laid side by side.

    Example
    -------
    >>> drawn = dense_graph(3, 2)
    >>> [box.name for box in drawn.boxes].count("edge")
    6
    >>> dense_graph(3, 2) is dense_graph(3, 2)
    True
    """
    return graph(tuple(complete(size) for _ in range(samples)), size)


@lru_cache(maxsize=16)
def pair_sites(size: int, samples: int) -> torch.Tensor:
    """
    Where the state of the pair ``(i, j)`` sits among the edge sites, as
    an ``(samples, n, n)`` index into a *padded* stack of them: entry
    ``0`` is a row of zeros, so the diagonal -- which is no box -- reads
    zero, and the pair ``(j, i)`` reads the same box as ``(i, j)``.

    That the ``e``-th edge site is the ``e``-th pair of
    :func:`~dataset.complete`, sample by sample, is a property of how
    :func:`~discopy.neural.from_incidence` orders its boxes;
    :func:`check_layout` checks it against the wiring rather than trusting
    it.

    Parameters:
        size : The number of nodes of every sample.
        samples : The number of samples laid side by side.

    Example
    -------
    >>> pair_sites(3, 1)
    tensor([[[0, 1, 2],
             [1, 0, 3],
             [2, 3, 0]]])
    """
    pairs = complete(size)
    grid = np.zeros((samples, size, size), dtype=np.int64)
    for sample in range(samples):
        found = 1 + sample * len(pairs) + np.arange(len(pairs))
        grid[sample, pairs[:, 0], pairs[:, 1]] = found
        grid[sample, pairs[:, 1], pairs[:, 0]] = found
    return torch.as_tensor(grid)


# --- the tensors a loss and a decode read ----------------------------------

@dataclass
class Batch:
    """
    A batch of trajectories run as one diagram, with the tensors its
    encoders, its losses and its metrics read.

    The site axis of every family is *global*, i.e. into the concatenation
    over the batch, and the node families reshape to ``(samples, nodes)``
    because every sample of a CLRS batch has the same ``n``.  That the
    ``i``-th node site is the ``i``-th node of the batch is a property of
    how :func:`~discopy.neural.from_incidence` orders its boxes rather than
    something declared anywhere, so :func:`check_layout` checks it.

    Parameters:
        algorithm : The algorithm the trajectories execute.
        diagram : Their incidence graph.
        size : The number of nodes of every sample.
        inputs : Per input probe, its array as CLRS produced it.
        hints : Per hint probe, the same with a leading step axis.
        outputs : Per output probe, the same.
        lengths : The number of steps of each trajectory.
        edge_inputs : Per edge feature, its value on every edge of the
                      batch, concatenated; see
                      :attr:`~dataset.Split.edge_inputs`.
        n_edges : The edge boxes of the whole batch.
        pairs : Where each pair's state sits among them, or ``None`` when
                the diagram is not the complete graph; see
                :func:`pair_sites`.
    """

    algorithm: str
    diagram: object
    size: int
    inputs: dict
    hints: dict
    outputs: dict
    lengths: torch.Tensor
    edge_inputs: dict
    n_edges: int = 0
    pairs: torch.Tensor = None

    @classmethod
    def of(cls, split: Split, diagram=None) -> Batch:
        """
        The batch of a :class:`~dataset.Split`, building its diagram unless
        one is given.

        Parameters:
            split : The trajectories.
            diagram : Their incidence graph, built here by default.
        """
        as_tensor = lambda value: torch.as_tensor(
            np.asarray(value), dtype=torch.float32)
        dense = split.algorithm in DENSE
        if diagram is None:
            diagram = dense_graph(split.n, len(split)) if dense \
                else graph(split.edges, split.n)
        given = {
            name: as_tensor(np.concatenate(
                [one[name] for one in split.edge_inputs] + [np.zeros(0)]))
            for name in edge_features(split.algorithm)}
        return cls(
            split.algorithm, diagram, split.n,
            {key: as_tensor(value) for key, value in split.inputs.items()},
            {key: as_tensor(value) for key, value in split.hints.items()},
            {key: as_tensor(value) for key, value in split.outputs.items()},
            torch.as_tensor(np.asarray(split.lengths), dtype=torch.long),
            given, sum(len(pairs) for pairs in split.edges),
            pair_sites(split.n, len(split)) if dense else None)

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def steps(self) -> int:
        """
        The algorithm steps the batch is run for: the longest trajectory
        of its own members, which is what the trajectory rule reads.

        The hint *axis* is longer -- it is the longest trajectory of the
        whole split, and every array carries it -- so this is read off the
        lengths and never off a shape.
        """
        return int(self.lengths.max())

    def to(self, device) -> Batch:
        """ The same batch with its tensors on a device. """
        move = lambda group: {key: value.to(device)
                              for key, value in group.items()}
        return Batch(
            self.algorithm, self.diagram, self.size, move(self.inputs),
            move(self.hints), move(self.outputs), self.lengths.to(device),
            move(self.edge_inputs), self.n_edges,
            None if self.pairs is None else self.pairs.to(device))


class Batches:
    """
    A split cut into batches, each one diagram, on a device.

    Building a batch draws a diagram (:func:`graph`) and running one
    compiles it (:meth:`~discopy.neural.MapNN.compile`); both are Python,
    and both happen once because the batching never changes.  The compiled
    interactions live in the model's own cache, keyed by the identity of
    the diagram, so the cache must hold a whole split or every epoch
    recompiles; :func:`build` takes the size.

    Parameters:
        split : The trajectories.
        size : The trajectories per batch.
        device : Where the tensors live.

    Example
    -------
    >>> from dataset import load
    >>> batches = Batches(load("bfs", "val"), 16)
    >>> len(batches), len(batches[0]), batches.samples
    (2, 16, 32)
    """
    def __init__(self, split: Split, size: int, device=None):
        self.split, self.size = split, size
        self.batches = [Batch.of(one) for one in split.batches(size)]
        if device is not None:
            self.batches = [one.to(device) for one in self.batches]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __getitem__(self, index):
        return self.batches[index]

    @property
    def samples(self) -> int:
        """ The trajectories of the whole split. """
        return sum(len(one) for one in self.batches)

    @classmethod
    def over(cls, splits, size: int, device=None) -> "Batches":
        """
        Several splits as one sequence of batches, each batch drawn from
        one split and therefore **homogeneous in ``n``**.

        This is what a mixed-size training set is here.  A batch is one
        compiled diagram, so a batch of mixed sizes would be a diagram
        per combination of sizes rather than one per size, and the
        compile cache would carry them all for no gain: what training on
        several sizes is *for* is putting several trajectory depths in
        the distribution, and a homogeneous batch does that as well as a
        ragged one.  The epoch shuffles across the batches, so a step
        still sees the sizes in a random order.

        Parameters:
            splits : The splits, in any order.
            size : The trajectories per batch.
            device : Where the tensors live.

        Example
        -------
        >>> from dataset import load
        >>> found = Batches.over([load("bfs", "val")] * 2, 16)
        >>> len(found), found.samples
        (4, 64)
        """
        found = cls(splits[0], size, device)
        for split in splits[1:]:
            found.batches = found.batches + cls(split, size, device).batches
        return found


def check_layout(model, batch: Batch) -> None:
    """
    Assert that the ``i``-th site of the node family is the ``i``-th node of
    the batch, that the readout relations come one per sample, and that the
    ``e``-th edge site is wired to the endpoints
    :func:`~model.pair_sites` says it is.

    Every decoder indexes the site axis by global node index and the edge
    decoders index a *pair* grid on top of that, and both are properties of
    the wiring rather than of anything declared, so both are checked
    against it.  This is the constructive half of the alignment discipline:
    :func:`alignment` pins the two clocks against each other, this pins the
    two layouts.

    Parameters:
        model : The model whose interpretation compiles the diagram.
        batch : The batch to check.
    """
    interaction = model.map.compile(batch.diagram)
    heads = interaction.heads["node", STATE]
    expected = len(batch) * batch.size
    assert len(heads) == expected, \
        f"{len(heads)} node sites for {expected} nodes"
    cmap = interaction.cmap
    for index, port in enumerate(heads):
        assert cmap.boxes[index].name == "node", \
            f"box {index} is a {cmap.boxes[index].name}, not a node"
        assert port in box_ports(cmap, index), \
            f"the {index}-th node state is not on the {index}-th box"
    assert interaction.sites(("readout", GSTATE)) == len(batch), \
        "there is not exactly one readout relation per sample"
    if batch.n_edges:
        if ("edge", ESTATE) in interaction.heads:
            assert interaction.sites(("edge", ESTATE)) == batch.n_edges, \
                "the edge sites do not match the edge list"
        check_edges(interaction, batch)
    if batch.pairs is not None:
        check_pairs(interaction, batch)


def endpoints(interaction) -> list:
    """
    Per edge site, the two node boxes its message ports reach, in port
    order.

    Parameters:
        interaction : The compiled diagram.
    """
    cmap = interaction.cmap
    owner = {port: box for box in range(len(cmap.boxes))
             for port in box_ports(cmap, box)}
    # the node-only arm has no edge state to enumerate the boxes by, and
    # the carried input is the family every edge keeps in both arms.
    family = ("edge", ESTATE) if ("edge", ESTATE) in interaction.heads \
        else ("edge", WEIGHT)
    return [tuple(owner[cmap.edges[port]]
                  for port in box_ports(cmap, owner[found])[:2])
            for found in interaction.heads[family]]


def check_edges(interaction, batch: Batch) -> None:
    """
    Assert that an edge box joins two nodes *of one sample* and that its
    **first** message port is the lower-indexed endpoint's.

    That order is what a directed cell reads its two legs by: the
    orientation of an edge is a bit on the carry, written by
    :attr:`~dataset.Split.edge_inputs` under the assumption that
    :func:`~discopy.neural.from_incidence` fills an edge's slots in node
    order.  It does, deterministically -- and were it ever not to, every
    direction of ``dag_shortest_paths`` would silently flip, which is a
    defect no loss curve would show.

    Parameters:
        interaction : The compiled diagram.
        batch : The batch to check.
    """
    for site, (one, other) in enumerate(endpoints(interaction)):
        assert one < other, \
            f"edge site {site} reads its endpoints as ({one}, {other}), " \
            f"so its first port is not the lower-indexed one"
        assert one // batch.size == other // batch.size, \
            f"edge site {site} joins two samples"


def check_pairs(interaction, batch: Batch) -> None:
    """
    Assert that the edge box the pair grid sends ``(i, j)`` to is the box
    wired to the ``i``-th and the ``j``-th node of that sample.

    Parameters:
        interaction : The compiled diagram.
        batch : The batch, whose ``pairs`` grid is being checked.
    """
    edges = endpoints(interaction)
    for sample in range(len(batch)):
        for one, other in complete(batch.size):
            site = int(batch.pairs[sample, one, other]) - 1
            assert edges[site] == (sample * batch.size + one,
                                   sample * batch.size + other), \
                f"the box of the pair ({one}, {other}) of sample " \
                f"{sample} is wired to {edges[site]}"


# --- 4. the feature types: encoders, decoders, losses ----------------------

class Encoder(torch.nn.Linear):
    """
    One input probe, embedded: a linear map from its scalar value to the
    message width, as CLRS encodes each input and sums the results.

    A ``mask``, a ``mask_one`` and a ``scalar`` all arrive as one number per
    node -- the benchmark's own dense encoding -- so one shape serves the
    three, and what distinguishes them is the *decoder* and the loss.

    Example
    -------
    >>> Encoder(4)(torch.zeros(2, 6, 1)).shape
    torch.Size([2, 6, 4])
    """
    def __init__(self, dim: int):
        super().__init__(1, dim)


@dataclass(frozen=True)
class Dims:
    """
    The widths a decoder reads.

    Parameters:
        node : The width of a node's state.
        edge : The width of a pair's state, ``0`` when the edges carry
               none -- which is H1's node-only arm, where every edge
               decoder falls back to the pair of node states alone.
        graph : The width of the readout relation's state.
        hidden : The width of the intermediate a pointer scores through.
        classes : The classes of a categorical probe.
    """
    node: int
    edge: int = 0
    graph: int = 0
    hidden: int = 16
    classes: int = 1


@dataclass
class Latents:
    """
    What one checkpoint offers a decoder: the state of every node, of
    every *pair* and of the graph.

    The pair grid is the reason Part 2 needed an edge box that remembers.
    A node probe is read off the node that answers it, but ``Pi[i, j]`` is
    a question about a *pair* -- and on the complete-graph diagram of
    :data:`~dataset.DENSE` every pair is a box, so the answer has somewhere
    to live.  It is ``None`` exactly when no edge carries a state, and
    every edge decoder then reads the two node states alone, which is what
    a plain message passer does.

    Parameters:
        nodes : ``(samples, n, node)``.
        pairs : ``(samples, n, n, edge)``, symmetric, zero on the diagonal.
        graph : ``(samples, graph)``.
    """
    nodes: torch.Tensor
    pairs: torch.Tensor = None
    graph: torch.Tensor = None


class ScalarDecoder(torch.nn.Module):
    """ One number per node, read off its state. """
    def __init__(self, dims: Dims):
        super().__init__()
        self.head = torch.nn.Linear(dims.node, 1)

    def forward(self, latents: Latents):
        return self.head(latents.nodes).squeeze(-1)

    @staticmethod
    def loss(prediction, truth):
        """ The mean squared error, CLRS's loss for a scalar probe. """
        return torch.nn.functional.mse_loss(prediction, truth)

    @staticmethod
    def score(prediction, truth):
        """ The mean squared error, CLRS's score for a scalar probe. """
        return float(((prediction - truth) ** 2).mean())


class MaskDecoder(ScalarDecoder):
    """ One logit per node, read off its state. """

    @staticmethod
    def loss(prediction, truth):
        """ The binary cross-entropy, CLRS's loss for a mask probe. """
        valid = truth != MASKED
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[valid], truth[valid])

    @staticmethod
    def score(prediction, truth):
        """
        The F1 of the thresholded logits, CLRS's ``_mask_fn`` verbatim:
        a mask probe is imbalanced, so accuracy would flatter it.
        """
        valid = np.asarray(truth) != MASKED
        found = (np.asarray(prediction) > 0) & valid
        true = (np.asarray(truth) > 0.5) & valid
        hit, miss = float((found & true).sum()), float((found & ~true).sum())
        gone = float((~found & true).sum())
        precision = hit / (hit + miss) if hit + miss > 0 else 1.0
        recall = hit / (hit + gone) if hit + gone > 0 else 1.0
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


class MaskOneDecoder(ScalarDecoder):
    """
    One logit per node, softmaxed over the nodes of a sample: the CLRS
    ``mask_one`` type is a categorical distribution over nodes.
    """

    @staticmethod
    def loss(prediction, truth):
        """ The softmax cross-entropy against the one-hot node. """
        return torch.nn.functional.cross_entropy(
            prediction, truth.argmax(-1))

    @staticmethod
    def score(prediction, truth):
        """ The accuracy of the argmax, CLRS's ``_eval_one``. """
        return float((np.asarray(prediction).argmax(-1)
                      == np.asarray(truth).argmax(-1)).mean())


class CategoricalDecoder(torch.nn.Module):
    """
    One logit per class per node: the ``color`` of ``dag_shortest_paths``
    is white, grey or black, and the benchmark one-hot encodes it.

    Example
    -------
    >>> CategoricalDecoder(Dims(node=4, classes=3))(
    ...     Latents(torch.zeros(2, 6, 4))).shape
    torch.Size([2, 6, 3])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.head = torch.nn.Linear(dims.node, dims.classes)

    def forward(self, latents: Latents):
        return self.head(latents.nodes)

    @staticmethod
    def loss(prediction, truth):
        """ The softmax cross-entropy against the one-hot class. """
        valid = (truth != MASKED).all(-1)
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        return torch.nn.functional.cross_entropy(
            prediction[valid], truth[valid].argmax(-1))

    @staticmethod
    def score(prediction, truth):
        """ The accuracy of the argmax, CLRS's ``_eval_one``. """
        valid = (np.asarray(truth) != MASKED).all(-1)
        return float((np.asarray(prediction).argmax(-1)
                      == np.asarray(truth).argmax(-1))[valid].mean())


class GraphMaskDecoder(torch.nn.Module):
    """
    One logit per *sample*, read off the readout relation's state: the
    ``phase`` of ``dag_shortest_paths``, which says whether the search is
    still ordering the nodes or already relaxing them.

    This is what the readout relation was kept stateful for.  A ``Relation``
    would pool the nodes into an answer and forget it; a graph-level probe
    is a property of the execution rather than of the graph, so it needs
    somewhere to accumulate -- see ``NOTES.md``.

    Example
    -------
    >>> GraphMaskDecoder(Dims(node=4, graph=8))(
    ...     Latents(torch.zeros(2, 6, 4), graph=torch.zeros(2, 8))).shape
    torch.Size([2])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.head = torch.nn.Linear(dims.graph, 1)

    def forward(self, latents: Latents):
        return self.head(latents.graph).squeeze(-1)

    loss = staticmethod(MaskDecoder.loss)
    score = staticmethod(MaskDecoder.score)


class PointerDecoder(torch.nn.Module):
    """
    A bilinear score over node-state pairs, softmaxed per row: node ``i``
    points at the node ``j`` maximising ``<W_1 h_i, W_2 h_j>``.

    Parameters:
        dims : The widths it reads.

    Example
    -------
    >>> PointerDecoder(Dims(node=4))(Latents(torch.zeros(2, 6, 4))).shape
    torch.Size([2, 6, 6])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.query = torch.nn.Linear(dims.node, dims.node)
        self.key = torch.nn.Linear(dims.node, dims.node)
        self.scale = dims.node ** -0.5

    def forward(self, latents: Latents):
        return self.scale * torch.einsum(
            "bid,bjd->bij", self.query(latents.nodes),
            self.key(latents.nodes))

    @staticmethod
    def loss(prediction, truth):
        """ The softmax cross-entropy of each row against its pointer. """
        return torch.nn.functional.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            truth.reshape(-1).long())

    @staticmethod
    def score(prediction, truth):
        """ The accuracy of the argmax, CLRS's score for a pointer. """
        return float((np.asarray(prediction).argmax(-1)
                      == np.asarray(truth)).mean())


class EdgePointerDecoder2(torch.nn.Module):
    """
    A node pointer scored from the **pair** as well as its two endpoints,
    which is ``clrs._src.decoders._decode_node_fts``' own form for a
    ``POINTER``::

        p_1, p_2 = source(h), target(h)          # (b, n, hidden)
        p_3      = pair(e)                       # (b, n, n, hidden)
        p_e      = p_2[..., None, :] + p_3
        p_m      = max(p_1[..., None, :], p_e.transpose)
        pred     = head(p_m)                     # (b, n, n)

    :class:`PointerDecoder` -- a bilinear score over node states alone --
    is what Part 2 measured with, and the difference is not a detail.  A
    pointer is the one probe type whose answer is *another node*, and the
    reference hands its decoder the edge channel at every step, so a
    parent can be scored from the edge that would carry it.  Ours could
    only score it from two node states, which is why the study's masks
    generalize and its pointers do not: this is M2's suspect, made
    runnable.

    On a sparse diagram the grid is zero where no box joins the pair (see
    :meth:`Model.latents`), so the edge term carries the adjacency the
    reference reads off ``adj``.  With no edge state at all the pair term
    is absent and this is :class:`PointerDecoder` with an MLP head, which
    keeps H1's node-only arm buildable.

    Example
    -------
    >>> nodes, pairs = torch.zeros(2, 6, 4), torch.zeros(2, 6, 6, 3)
    >>> EdgePointerDecoder2(Dims(node=4, edge=3, hidden=5))(
    ...     Latents(nodes, pairs)).shape
    torch.Size([2, 6, 6])
    >>> EdgePointerDecoder2(Dims(node=4, hidden=5))(Latents(nodes)).shape
    torch.Size([2, 6, 6])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        hidden = dims.hidden or dims.node
        self.source = torch.nn.Linear(dims.node, hidden)
        self.target = torch.nn.Linear(dims.node, hidden)
        self.pair = torch.nn.Linear(dims.edge, hidden) if dims.edge else None
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, latents: Latents):
        source = self.source(latents.nodes).unsqueeze(-2)
        target = self.target(latents.nodes).unsqueeze(-2)
        if self.pair is not None and latents.pairs is not None:
            target = target + self.pair(latents.pairs)
        return self.head(
            torch.maximum(source, target.transpose(-3, -2))).squeeze(-1)

    loss = staticmethod(PointerDecoder.loss)
    score = staticmethod(PointerDecoder.score)


class EdgeBilinearPointerDecoder(torch.nn.Module):
    """
    The edge channel of :class:`EdgePointerDecoder2` with the *combiner*
    of :class:`PointerDecoder`: a bilinear score over node states plus a
    linear read of the pair's state.

    :class:`EdgePointerDecoder2` changes two things at once against the
    study's original head -- it adds the edge term **and** it scores
    through a ``max`` of broadcast terms rather than a dot product -- so
    a gain from it does not say which.  Those are different mechanisms:
    one is an extra input channel, the other is an order statistic in the
    head.  This is the third corner that separates them.

    Example
    -------
    >>> nodes, pairs = torch.zeros(2, 6, 4), torch.zeros(2, 6, 6, 3)
    >>> EdgeBilinearPointerDecoder(Dims(node=4, edge=3))(
    ...     Latents(nodes, pairs)).shape
    torch.Size([2, 6, 6])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.query = torch.nn.Linear(dims.node, dims.node)
        self.key = torch.nn.Linear(dims.node, dims.node)
        self.pair = torch.nn.Linear(dims.edge, 1) if dims.edge else None
        self.scale = dims.node ** -0.5

    def forward(self, latents: Latents):
        found = self.scale * torch.einsum(
            "bid,bjd->bij", self.query(latents.nodes),
            self.key(latents.nodes))
        if self.pair is not None and latents.pairs is not None:
            found = found + self.pair(latents.pairs).squeeze(-1)
        return found

    loss = staticmethod(PointerDecoder.loss)
    score = staticmethod(PointerDecoder.score)


class EdgeScalarDecoder(torch.nn.Module):
    """
    One number per **pair**, read off the pair's state and the states of
    its two endpoints:
    ``pred[i, j] = w_1 h_i + w_2 h_j + w_e e_ij``.

    The three terms are ``clrs._src.decoders._decode_edge_fts``' own, and
    the last one is the whole of H1: erase the edge state and the head
    still answers, from an ordered pair of node states and nothing else.
    That the edge term is the *only* difference between the two arms is
    what makes their comparison one axis.

    Example
    -------
    >>> nodes, pairs = torch.zeros(2, 6, 4), torch.zeros(2, 6, 6, 3)
    >>> EdgeScalarDecoder(Dims(node=4, edge=3))(
    ...     Latents(nodes, pairs)).shape
    torch.Size([2, 6, 6])
    >>> EdgeScalarDecoder(Dims(node=4))(Latents(nodes)).shape
    torch.Size([2, 6, 6])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.source = torch.nn.Linear(dims.node, 1)
        self.target = torch.nn.Linear(dims.node, 1)
        self.pair = torch.nn.Linear(dims.edge, 1) if dims.edge else None

    def forward(self, latents: Latents):
        found = self.source(latents.nodes) \
            + self.target(latents.nodes).transpose(-1, -2)
        if self.pair is not None:
            found = found + self.pair(latents.pairs).squeeze(-1)
        return found

    loss = staticmethod(ScalarDecoder.loss)
    score = staticmethod(ScalarDecoder.score)


class EdgeMaskDecoder(EdgeScalarDecoder):
    """ One logit per pair: the ``msk`` of the two dynamic programs. """

    loss = staticmethod(MaskDecoder.loss)
    score = staticmethod(MaskDecoder.score)


class EdgePointerDecoder(torch.nn.Module):
    """
    A pointer per **pair**: ``Pi[i, j]`` is the node the shortest path from
    ``i`` to ``j`` arrives from, ``s[i, j]`` the index an interval splits
    at.

    ``clrs._src.decoders._decode_edge_fts``' shape, verbatim: the pair
    embedding is broadcast against every candidate node, the two are
    combined by a maximum -- the one order-independent choice that keeps
    the score of a candidate from being diluted by the others -- and a
    linear head reads one logit per ``(i, j, k)``.

    Example
    -------
    >>> nodes, pairs = torch.zeros(2, 6, 4), torch.zeros(2, 6, 6, 3)
    >>> EdgePointerDecoder(Dims(node=4, edge=3, hidden=5))(
    ...     Latents(nodes, pairs)).shape
    torch.Size([2, 6, 6, 6])
    """
    def __init__(self, dims: Dims):
        super().__init__()
        self.source = torch.nn.Linear(dims.node, dims.hidden)
        self.target = torch.nn.Linear(dims.node, dims.hidden)
        self.pair = torch.nn.Linear(dims.edge, dims.hidden) \
            if dims.edge else None
        self.candidate = torch.nn.Linear(dims.node, dims.hidden)
        self.head = torch.nn.Linear(dims.hidden, 1)

    def forward(self, latents: Latents):
        found = self.source(latents.nodes).unsqueeze(-2) \
            + self.target(latents.nodes).unsqueeze(-3)
        if self.pair is not None:
            found = found + self.pair(latents.pairs)
        return self.head(torch.maximum(
            found.unsqueeze(-2),
            self.candidate(latents.nodes).unsqueeze(-3).unsqueeze(-3)
        )).squeeze(-1)

    loss = staticmethod(PointerDecoder.loss)
    score = staticmethod(PointerDecoder.score)


#: The decoder *class* of each ``(location, type)`` the eight algorithms
#: use.  A class, not an instance: a probe is decoded by a decoder of its
#: own, so what a feature type fixes is the shape of the head and its
#: loss, not the weights.  Part 1 had the four node rows; the edge rows
#: are what ``floyd_warshall`` and ``matrix_chain_order`` needed and what
#: H1 is measured through, and the last two belong to
#: ``dag_shortest_paths`` alone.
#: The node-pointer heads a run may build, a key of
#: :attr:`~config.Budget.pointer`.  ``"bilinear"`` is what Part 2
#: measured with, ``"edge"`` is the reference's own form and
#: ``"edgebilinear"`` is the corner that separates its two changes.
POINTERS = {
    "bilinear": None,
    "edge": EdgePointerDecoder2,
    "edgebilinear": EdgeBilinearPointerDecoder,
}

DECODERS = {
    ("node", "scalar"): ScalarDecoder,
    ("node", "mask"): MaskDecoder,
    ("node", "mask_one"): MaskOneDecoder,
    ("node", "pointer"): PointerDecoder,
    ("node", "categorical"): CategoricalDecoder,
    ("edge", "scalar"): EdgeScalarDecoder,
    ("edge", "mask"): EdgeMaskDecoder,
    ("edge", "pointer"): EdgePointerDecoder,
    ("graph", "mask"): GraphMaskDecoder,
}


def decoded(algorithm: str) -> tuple:
    """
    The probes one algorithm decodes -- every hint and every output -- in
    the spec's order, which is the order their decoders are built in.

    **One decoder per probe, not per feature type.**  Two probes of the
    same type are two different questions asked of the same state, and a
    head answers one: ``minimum`` decodes ``min``, ``min_h`` and ``i`` as
    three ``node/mask_one`` probes, and they are respectively the answer,
    the running answer and the loop counter.  Sharing a head between them
    asks one logit per node to be three distributions at once, which no
    weights can be.  It is also what CLRS does --
    ``clrs._src.nets.Net._construct_encoders_decoders`` keys ``dec`` by
    ``name``.

    Parameters:
        algorithm : The algorithm.

    Example
    -------
    >>> decoded("bellman_ford")
    ('pi_h', 'd', 'msk', 'pi')
    >>> decoded("minimum")
    ('pred_h', 'min_h', 'i', 'min')
    """
    return tuple(name for stage in ("hint", "output")
                 for name in probes(algorithm, stage))


def encoded(algorithm: str) -> tuple:
    """
    What is written into the initial state, in encoder order: the node
    input probes on the node input loop, then what every edge box is
    given on the edge weight loop.

    ``adj`` is among the latter for a :data:`~dataset.DENSE` algorithm and
    for no other.  Everywhere else an edge box exists exactly where ``adj``
    is true, so the diagram already says everything ``adj`` says and
    encoding it would write a learned constant; on the complete-graph
    diagram it says nothing of the kind, and the sampled graph has to be
    told.  ``orient`` is the same story for the one directed algorithm:
    a pair is one box, so which way it points is an input rather than a
    wiring.

    Parameters:
        algorithm : The algorithm.

    Example
    -------
    >>> encoded("bfs")
    (('pos', 'node'), ('s', 'node'), ('A', 'edge'))
    >>> encoded("floyd_warshall")
    (('pos', 'node'), ('A', 'edge'), ('adj', 'edge'))
    >>> encoded("dag_shortest_paths")[-2:]
    (('A', 'edge'), ('orient', 'edge'))
    """
    return tuple(
        (name, "node") for name in probes(algorithm, "input")
        if kind(algorithm, name)[0] == "node") + tuple(
        (name, "edge") for name in edge_features(algorithm))


# --- the model -------------------------------------------------------------

class Model(torch.nn.Module):
    """
    A learned algorithm: an encoder per input, a
    :class:`~discopy.neural.MapNN` and one decoder per decoded probe.

    Parameters:
        algorithm : The algorithm being imitated.
        interpretation : The :class:`~discopy.neural.MapNN`.
        encoders : One :class:`Encoder` per encoded input probe.
        decoders : One decoder per decoded probe; see :func:`decoded`.
        hops : The rounds of message passing one algorithm step costs.
        steps : The algorithm steps a run covers, or ``None`` for the
                trajectory rule -- as many steps as the batch's longest
                trajectory takes.  A number is Part 1's fixed depth, kept
                so that its row stays reproducible.
        hint_weight : The weight of the per-step hint loss.
        settle : How a finished trajectory's remaining checkpoints are
                 supervised, a member of :data:`config.SETTLE`; see
                 :meth:`Model.hint_targets`.
        probe : Whether the hint loss is decoded from a detached state,
                so that it fits the hint decoders and never the
                interaction: Part 3's output-only arms; see
                :meth:`Model.loss`.
    """
    def __init__(self, algorithm: str, interpretation: MapNN,
                 encoders: torch.nn.ModuleDict,
                 decoders: torch.nn.ModuleDict, hops: int = HOPS,
                 steps: int = None, hint_weight: float = HINT_WEIGHT,
                 settle: str = None, probe: bool = False):
        super().__init__()
        self.algorithm = algorithm
        self.map = interpretation
        self.encoders = encoders
        self.decoders = decoders
        self.hops, self.steps = hops, steps
        self.hint_weight = hint_weight
        self.settle = holding(settle)
        self.probe = probe
        self.answer = ("node", STATE)

    # --- what the model needs to know about the map ------------------------

    def steps_of(self, batch: Batch) -> int:
        """ The algorithm steps a run on this batch covers. """
        return batch.steps if self.steps is None else self.steps

    def rounds_for(self, batch: Batch, factor: float = 1.0) -> int:
        """
        The rounds one run on a batch costs, and the only place a depth is
        decided.

        The test-time sweep is a factor on the *steps* rather than on the
        rounds, so every point of it stays a whole number of hops and its
        checkpoints keep landing where :func:`alignment` says they do.

        Parameters:
            batch : The batch to run.
            factor : The multiple of the trained depth to run at.
        """
        return rounds_of(
            max(1, int(round(factor * self.steps_of(batch)))), self.hops)

    # --- the two ends of the task ------------------------------------------

    def encode(self, batch: Batch, values: dict) -> dict:
        """
        The initial values of every written family: the summed embeddings
        of the node inputs on the node input loop, of the edge inputs on
        the edge weight loop.

        Parameters:
            batch : The batch to run.
            values : The tensors to write, added to in place.
        """
        node_input, edge_input = None, None
        for name, location in encoded(self.algorithm):
            encoder = self.encoders[name]
            if location == "node":
                found = encoder(batch.inputs[name].reshape(-1, 1))
                node_input = found if node_input is None \
                    else node_input + found
            elif batch.n_edges:
                found = encoder(batch.edge_inputs[name].reshape(-1, 1))
                edge_input = found if edge_input is None \
                    else edge_input + found
        if node_input is not None:
            values["node", FEAT] = node_input.unsqueeze(0)
        if edge_input is not None:
            values["edge", WEIGHT] = edge_input.unsqueeze(0)
        return values

    def states(self, batch: Batch, state):
        """ The node states of a batch, of shape ``(samples, nodes, w)``. """
        return self.map.read(batch.diagram, state, self.answer).reshape(
            len(batch), batch.size, -1)

    def latents(self, batch: Batch, state) -> Latents:
        """
        What a checkpoint offers its decoders: the node states, the pair
        grid when the edges carry a state, and the graph state.

        Parameters:
            batch : The batch the state belongs to.
            state : The flat messages of a checkpoint.
        """
        interaction = self.map.compile(batch.diagram)
        pairs = None
        if batch.pairs is not None and ("edge", ESTATE) in interaction.heads:
            found = interaction.read(state, ("edge", ESTATE))
            padded = torch.cat([torch.zeros_like(found[:, :1]), found], 1)
            pairs = padded[:, batch.pairs].reshape(
                len(batch), batch.size, batch.size, -1)
        return Latents(
            self.states(batch, state), pairs,
            interaction.read(state, ("readout", GSTATE)).reshape(
                len(batch), -1))

    def decode(self, batch: Batch, state, names=None) -> dict:
        """
        Every probe the algorithm decodes, read off one checkpoint.

        Parameters:
            batch : The batch the state belongs to.
            state : The flat messages of a checkpoint.
            names : The probes to decode, all of them by default.

        Returns:
            The prediction of each probe.
        """
        latents = self.latents(batch, state)
        return {name: self.decoders[name](latents)
                for name in (decoded(self.algorithm) if names is None
                             else names)}

    # --- running it --------------------------------------------------------

    def initial(self, batch: Batch, rows: int = 1):
        """ The initial flat state: the encoded inputs, zeros elsewhere. """
        return self.map.initial(
            batch.diagram, self.encode(batch, {}), rows=rows)

    def checkpoints(self, every: list) -> list:
        """
        The states that correspond to a step of the algorithm, i.e. the
        rounds :func:`alignment` names.

        Parameters:
            every : The state after every round.
        """
        return [every[found - 1]
                for _, found, _ in alignment(len(every), self.hops)]

    def run(self, batch: Batch, state=None, deep: bool = False,
            factor: float = 1.0, **overrides) -> list:
        """
        The states a run passes through: one per algorithm step under
        ``deep``, the last one alone otherwise.

        Parameters:
            batch : The batch to run.
            state : The flat initial messages, built here by default.
            deep : Whether to keep every checkpoint.
            factor : The multiple of the trained depth to run at.
            overrides : Test-time compute, i.e. an explicit ``rounds``.
        """
        overrides.setdefault("rounds", self.rounds_for(batch, factor))
        state = self.initial(batch) if state is None else state
        if not deep:
            return [self.map(batch.diagram, state, **overrides)]
        return self.checkpoints(
            self.map(batch.diagram, state, deep=True, **overrides))

    def forward(self, batch: Batch, state=None, deep: bool = False,
                names=None, **overrides):
        """
        The decoded predictions of a whole run: one dictionary per
        algorithm step under ``deep``, the last one otherwise.

        The two agree exactly when the rounds run are a multiple of
        :attr:`hops`, which every protocol here keeps them: a deep run
        reads its checkpoints at hop boundaries, so a shallow run of an odd
        number of rounds ends half a hop past the last of them.

        Parameters:
            batch : The batch to run.
            state : The flat initial messages, built here by default.
            deep : Whether to return every step rather than the last.
            names : The probes to decode, all of them by default.
            overrides : Test-time compute, i.e. ``factor`` or ``rounds``.
        """
        every = self.run(batch, state, deep=deep, **overrides)
        found = [self.decode(batch, one, names) for one in every]
        return found if deep else found[-1]

    # --- the loss ----------------------------------------------------------

    def hint_targets(self, batch: Batch, step: int, settle=None) -> dict:
        """
        The hint of one algorithm step, grouped by feature type, together
        with the trajectories the step is defined for.

        A trajectory of ``length`` steps defines ``hints[0] ... hints[
        length - 1]``, and ``hints[0]`` is the initial condition rather
        than something to predict, so the ``step``-th checkpoint is
        supervised on ``hints[step + 1]`` where that exists -- the hint
        column of :func:`alignment`, which is where the index comes from.

        Past that, ``settle`` decides what a finished trajectory is worth,
        and it is a member of :data:`config.SETTLE` rather than a flag
        because the two ways of holding a hint are not the same
        experiment.

        ``None`` drops it, which is the protocol every number in Part 2
        was measured under: a sample supervises nothing after its own last
        step, so what the map does with a converged state was never in the
        loss.  ``"interior"`` clamps its index at its own last step, so
        the extra checkpoints are supervised on the final hint repeated.
        A converged algorithm emits a constant hint, and holding it is the
        difference between a map that has reached the answer and one that
        has reached it *and stays there* -- which is a basin at
        termination, i.e. exactly the property a
        :class:`~discopy.neural.FixedPoint` goes looking for.  The output
        loss has always been supervised this way (see :meth:`loss`, "reach
        the answer and then stay there"); this makes the hints agree with
        it.

        ``"interior"`` reaches every checkpoint **but the last one**, and
        that exception is the reason for the third member.  Under the
        trajectory rule a run is ``batch.steps`` checkpoints long and the
        ``k``-th is supervised on hint ``k + 1``, so the last one asks for
        hint ``batch.steps`` -- which the guard below refuses before
        ``settle`` is consulted at all.  Measured on ``bfs`` at
        ``n = 16``: the last checkpoint of a batch is supervised on **no
        rows** under either setting, while ``"interior"`` lifts the one
        before it from 8 rows of 32 to all 32.  So a hold that stops short
        of termination trains a basin everywhere except at the state a
        fixed point converges to, which is the one place H2 reads.
        ``"terminal"`` holds it there too, and is what Part 3 trains with;
        ``"interior"`` is kept because it is what the salvaged mixed
        campaign was trained under and a tag has to keep meaning what it
        meant.

        Parameters:
            batch : The batch.
            step : The zero-based checkpoint, i.e. the algorithm step.
            settle : How to hold a finished trajectory's last hint, the
                     model's own setting by default.  Scoring passes
                     override it to ``False`` so that a hint curve means
                     one thing across campaigns: a model trained to hold
                     its answer would otherwise be measured on the held
                     steps too, and its curve would not be comparable
                     with a curve from a model that was not.

        Example
        -------
        >>> import dataset
        >>> batch = Batch.of(dataset.load("bfs", "train").subsample(32))
        >>> model = build("bfs", Widths(8, 16, 16, 8, 16))
        >>> def rows(step, settle):
        ...     found = model.hint_targets(batch, step, settle)
        ...     return int(found["reach_h"][1].sum()) if found else 0
        >>> last = batch.steps - 1
        >>> rows(last, None), rows(last, "interior"), rows(last, "terminal")
        (0, 0, 32)
        >>> rows(last - 1, None), rows(last - 1, "interior")
        (8, 32)
        """
        index = hint_of(step)
        settle = holding(self.settle if settle is None else settle)
        if index >= batch.steps and settle != "terminal":
            return {}
        alive = index < batch.lengths
        if settle and bool((~alive).any()):
            clamped = torch.minimum(
                torch.full_like(batch.lengths, index), batch.lengths - 1)
            rows = torch.arange(len(batch), device=clamped.device)
            everyone = torch.ones_like(alive)
            return {name: (batch.hints[name][clamped, rows], everyone)
                    for name in probes(self.algorithm, "hint")}
        if not bool(alive.any()):
            return {}
        return {name: (batch.hints[name][index][alive], alive)
                for name in probes(self.algorithm, "hint")}

    def loss(self, batch: Batch, state=None, **overrides):
        """
        The supervised loss of one run: the output loss on every checkpoint
        from the trajectory's last step onwards, plus the hint loss on
        every step the trajectory defines.

        Supervising the output *from the end of the trajectory onwards*
        rather than at the last round alone is the protocol choice that
        makes the test-time depth sweep meaningful: it asks the model to
        reach the answer and then to stay there, which is what depth
        robustness is.

        Under the trajectory rule the run is exactly as long as the longest
        trajectory of the batch, so ``settled`` never clamps and a sample
        that finishes early is supervised on its answer at every checkpoint
        after it -- which is what Part 1's fixed depth could not do for an
        algorithm whose trajectory was longer than the run, and why its
        ``minimum`` scored 1.00 at the depth it was trained at and 0.78 at
        three times that.

        The parts it reports are **per probe** as well as per stage. The
        total is what is optimised and the stages are what a curve is
        usually drawn from, but neither can show a head that is failing
        while its neighbours carry the sum -- which is precisely how a
        decoder shared between `minimum`'s three ``mask_one`` probes
        survived a whole campaign; see ``NOTES.md``. Part 2's probes are
        many more and its edge-level hints are matrices, so the per-probe
        term is also the hint-divergence diagnostic that part asks for.

        Under :attr:`probe` the hint half is decoded from a **detached**
        state, so it reaches the hint decoders and not the interaction.
        That is what an output-only arm of Part 3 means, and it means it
        for a reason a bare ``hint_weight = 0`` cannot serve: with no hint
        term at all the hint decoders receive no gradient, so the curves
        the per-head split is read from would be produced by untrained
        heads and an arm would score zero on the order-free column
        without its processor having failed at anything.  Detaching keeps
        the supervision regime honest -- the interaction is trained on the
        output alone, which is the axis -- and turns the hint heads into
        what they then are, linear probes of the state.  Their numbers
        carry that caveat and ``PART3.md`` states it.

        Parameters:
            batch : The batch to run.
            state : The flat initial messages, built here by default.
            overrides : Test-time compute, i.e. ``rounds``.

        Returns:
            The loss, and its parts: ``output``, ``hint`` and one
            ``probe/<name>`` per decoded probe.
        """
        every = self.run(batch, state, deep=True, **overrides)
        settled = (batch.lengths - 1).clamp(max=len(every) - 1)
        output, hint, each = 0.0, 0.0, {}
        for step, found in enumerate(every):
            done = settled <= step
            if bool(done.any()):
                names = probes(self.algorithm, "output")
                prediction = self.decode(batch, found, names=names)
                for name in names:
                    term = self.decoders[name].loss(
                        prediction[name][done], batch.outputs[name][done])
                    output, each[name] = output + term, each.get(
                        name, 0.0) + term
            targets = self.hint_targets(batch, step)
            if targets:
                prediction = self.decode(
                    batch, found.detach() if self.probe else found,
                    names=list(targets))
                for name, (truth, alive) in targets.items():
                    term = self.decoders[name].loss(
                        prediction[name][alive], truth)
                    hint, each[name] = hint + term, each.get(name, 0.0) + term
        steps = max(len(every), 1)
        output, hint = output / steps, hint / steps
        return output + self.hint_weight * hint, {
            "output": _number(output), "hint": _number(hint),
            **{f"probe/{name}": _number(term / steps)
               for name, term in each.items()}}

    @torch.no_grad()
    def predict(self, batch: Batch, **overrides) -> dict:
        """
        Per output probe, the prediction and the ground truth, as arrays.

        The pair rather than a score, because the scores of a split are not
        the mean of the scores of its batches -- an F1 is not linear -- so
        a caller pools these and scores once; see :func:`score`.

        Parameters:
            batch : The batch to run.
            overrides : Test-time compute, i.e. ``rounds``.
        """
        names = probes(self.algorithm, "output")
        prediction = self(batch, names=names, **overrides)
        return {name: (prediction[name].cpu().numpy(),
                       batch.outputs[name].cpu().numpy())
                for name in names}

    def evaluate(self, batch: Batch, **overrides) -> dict:
        """ The score of one batch; see :func:`score`. """
        return score(self.algorithm, self.predict(batch, **overrides))


def score(algorithm: str, found: dict) -> dict:
    """
    CLRS's own score of every output probe, and their mean.

    The per-type scores are :mod:`clrs._src.evaluation`'s verbatim --
    accuracy for a pointer and a ``mask_one``, F1 for a ``mask``, mean
    squared error for a scalar -- and the mean over output probes is what
    that module calls ``score`` and the papers report as micro-F1.

    Parameters:
        algorithm : The algorithm.
        found : Per output probe, the ``(prediction, truth)`` arrays.
    """
    scores = {name: DECODERS[kind(algorithm, name)].score(*pair)
              for name, pair in found.items()}
    scores["score"] = sum(scores.values()) / len(scores)
    return scores


def predictions(model, batches: Batches, **overrides) -> dict:
    """
    Every output probe of a whole split, prediction beside truth.

    Parameters:
        model : The trained model.
        batches : The :class:`Batches` to run.
        overrides : Test-time compute, i.e. ``rounds``.
    """
    model.eval()
    pooled: dict = {}
    for batch in batches:
        for name, pair in model.predict(batch, **overrides).items():
            pooled.setdefault(name, ([], []))
            for stack, value in zip(pooled[name], pair):
                stack.append(value)
    return {name: tuple(np.concatenate(stack) for stack in pair)
            for name, pair in pooled.items()}


def evaluate_split(model, batches: Batches, **overrides) -> dict:
    """
    The score of a whole split, pooled over its batches rather than
    averaged over them, because an F1 is not linear.

    Parameters:
        model : The trained model.
        batches : The :class:`Batches` to score.
        overrides : Test-time compute, i.e. ``rounds``.
    """
    return score(model.algorithm, predictions(model, batches, **overrides))


def _number(value) -> float:
    """ A python float out of a loss term, whether or not it is a tensor. """
    return float(value.detach()) if isinstance(value, torch.Tensor) \
        else float(value)


# --- building one ----------------------------------------------------------

def has_edges(algorithm: str) -> bool:
    """ Whether an algorithm's diagram has edge boxes at all. """
    return algorithm in DENSE or "adj" in SPECS[algorithm]


def build(algorithm: str, widths: Widths = None, steps: int = None,
          pool: str = POOL, edge_state: bool = True, per_leg: bool = True,
          cache: int = 256, hint_weight: float = HINT_WEIGHT,
          settle: str = None, pointer: str = "bilinear",
          solver: str = "iterate", backward: str = "full",
          probe: bool = False) -> Model:
    """
    The message-passing model: a diagram per batch, one shared cell per
    generator name, one encoder per input and one decoder per probe.

    Parameters:
        algorithm : The algorithm to imitate.
        widths : The widths, ``WIDTHS["mpnn"]`` by default.
        steps : The algorithm steps a run covers, ``None`` for the
                trajectory rule; see :meth:`Model.rounds_for`.
        pool : The key in :data:`~discopy.neural.cells.POOL` every site
               reduces its message orbit with.  It is the one hyper
               -parameter here that a *change of size* can see: a mean and
               a sum both rescale when a node's degree grows from ``n = 16``
               to ``n = 64``, a max does not.
        edge_state : Whether an edge box carries a state of its own.
                     ``False`` sends ``ESTATE`` to ``Dim(0)``, which erases
                     the state ports from the same diagram: H1's node-only
                     arm.
        per_leg : Whether a site answers each of its legs separately,
                  which is strictly more general than broadcasting one
                  belief and still permutation equivariant.
        cache : How many compiled diagrams the interpretation keeps; a
                whole epoch's batches should fit, or every epoch pays for
                its compilation again.
        hint_weight : The weight of the per-step hint loss.
        settle : How a finished trajectory's remaining checkpoints are
                 supervised, a member of :data:`config.SETTLE`; see
                 :meth:`Model.hint_targets`.
        pointer : Which node-pointer head to build, ``"bilinear"`` for
                  :class:`PointerDecoder` -- what Part 2 measured with --
                  or ``"edge"`` for :class:`EdgePointerDecoder2`, the
                  reference's own form and the head Part 3 freezes.
        solver : The execution and differentiation policy, a key of
                 :data:`SOLVERS`.
        backward : ``"full"`` or ``"last"``, read by a fixed-point
                   solver alone.
        probe : Whether the hint loss reaches the decoders alone; see
                :meth:`Model.loss`.

    Example
    -------
    >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
    <torch...>
    >>> count_parameters(build("bfs", Widths(8, 16, 16, 8, 16))) > 0
    True
    >>> build("bfs", Widths(8, 16, 16, 8, 16),
    ...       solver="fixedpoint", backward="last").map.solver
    FixedPoint()
    """
    widths = widths or WIDTHS["mpnn"]
    ob = graph_ob(widths, None if edge_state else 0)
    sizes = widths_of(ob)

    site = Site(node(), sizes, {STATE: Mode.STATE, FEAT: Mode.INPUT},
                hidden=widths.hidden, depth=2, pool=pool, recurrent="gru",
                emit=True, per_leg=per_leg, resumable=True)
    link = Link(edge(algorithm in DIRECTED), sizes, hidden=widths.hidden,
                depth=2, pool=pool, recurrent="gru") \
        if has_edges(algorithm) else None
    readout_site = Site(
        readout(), sizes, {GSTATE: Mode.STATE}, hidden=widths.hidden,
        depth=2, pool=pool, recurrent="gru", emit=True, per_leg=per_leg)
    encoders = torch.nn.ModuleDict({
        name: Encoder(widths.dim) for name, _ in encoded(algorithm)})
    heads = dict(DECODERS)
    if pointer != "bilinear":
        heads[("node", "pointer")] = POINTERS[pointer]
    decoders = torch.nn.ModuleDict({
        name: heads[kind(algorithm, name)](Dims(
            node=widths.state_dim, edge=sizes[ESTATE], graph=widths.graph_dim,
            hidden=widths.dim, classes=CLASSES.get((algorithm, name), 1)))
        for name in decoded(algorithm)})

    ar = {"node": site, "readout": readout_site}
    if link is not None:
        ar["edge"] = link
    carried = (("node", FEAT), ) + (
        (("edge", WEIGHT), ) if link is not None else ())
    return Model(
        algorithm,
        MapNN(ob, ar, solver=SOLVERS[solver](backward, carried), cache=cache),
        encoders, decoders, steps=steps, hint_weight=hint_weight,
        settle=settle, probe=probe)


def fit_cache(model, *batches) -> int:
    """
    Enlarge the interpretation's compile cache until every diagram that
    will be run fits in it at once, and return the size.

    A batch is a diagram and a diagram is compiled once, so the whole cost
    of compilation is paid in the first epoch and nothing after it --
    *provided the cache holds them all*.  A cache one diagram too small
    turns that one-off into a per-epoch cost, silently, because an evicting
    LRU and a warm one differ in wall clock and in nothing else.  Sizing it
    from the batches that exist rather than from an arithmetic guess is
    what makes the difference impossible; :meth:`MapNN.cache_stats` is what
    makes it visible.

    Parameters:
        model : The model whose interpretation to resize.
        batches : The :class:`Batches` that will be run.
    """
    model.map.cache = max(model.map.cache, sum(map(len, batches)))
    return model.map.cache


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def matched(algorithm: str, widths: Widths = None, span: range = None) -> int:
    """
    The hidden width at which H1's node-only arm has as many parameters as
    its edge-state arm.

    An edge cell without a recurrent state is a *smaller* cell in one place
    and a larger one in another -- it loses a GRU and gains an emission
    that reads a hidden vector instead of a state -- so the two arms do not
    match by default, and comparing them as they come would compare memory
    against capacity.  This searches the one width that is free to move,
    and :data:`config.WIDTHS`'s ``paired`` is the recorded answer rather
    than a number someone liked.

    Parameters:
        algorithm : The algorithm to match on.
        widths : The widths of the edge-state arm.
        span : The hidden widths to search.

    Example
    -------
    >>> matched("floyd_warshall", Widths(8, 16, 16, 8, 16),
    ...         span=range(8, 24))
    19
    """
    widths = widths or WIDTHS["mpnn"]
    reference = count_parameters(build(algorithm, widths))
    span = span or range(widths.hidden // 2, 2 * widths.hidden)
    return min(span, key=lambda hidden: abs(reference - count_parameters(
        build(algorithm, replace(widths, hidden=hidden), edge_state=False))))


def calls_per_round(model, batch: Batch) -> int:
    """
    The batched module calls one round of message passing costs, i.e. the
    number of distinct ``(name, port signature)`` groups of the compiled
    diagram.

    Degree heterogeneity is what this measures: every site of one name and
    one degree is evaluated in a single call, so a batch of graphs whose
    degrees spread widely pays more launches for the same arithmetic.

    Parameters:
        model : The model whose interpretation compiles the diagram.
        batch : The batch to compile.
    """
    cmap = model.map.compile(batch.diagram).cmap
    return len({(box.name, box.dom, box.cod) for box in cmap.boxes})


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
