# -*- coding: utf-8 -*-

"""
How a global interaction is executed.

:class:`~discopy.neural.map.Interaction` says what one round is,
:math:`T_{D,\\theta} = \\sigma_D \\circ \\Phi_\\theta`.  A solver says how
many rounds to run, when to read a checkpoint off the state and which of
those rounds are in the autograd graph.  That is *two* policies -- an
execution policy and a differentiation policy -- and naming them together
is deliberate: they are what distinguishes the architectures of the
literature, which otherwise share a wiring and a cell.

Four strategies, from the least to the most machinery:

* :class:`Iterate` -- the finite iterate :math:`T^n`, optionally with a
  checkpoint after every round so that a loss can supervise all of them in
  one backward graph.  A geometry-of-interaction factor graph is
  ``Iterate(rounds=16)``.
* :class:`FixedPoint` -- the same iteration run *towards* a fixed point
  :math:`s^\\ast = T(s^\\ast)`, stopping on the residual rather than on a
  round count.  Nothing guarantees it converges: contractivity is an
  analytic property of the learned weights, which
  :meth:`~discopy.neural.map.Interaction.residual` measures and the
  category never supplies.
* :class:`Recursion` -- the segmented recursion of
  :cite:t:`JolicoeurMartineau25`: ``steps`` supervision steps detached from
  one another, each ``cycles`` cycles of ``rounds`` rounds, only the last
  cycle differentiated, with an optional :class:`Refresh` of a trace
  between cycles.  Its activation memory is set by one segment rather than
  by the depth run.
* :class:`ACT` -- a recursion with a :class:`HaltHead`, i.e. adaptive
  computation time.

Resuming is a property of the formalism, :math:`T^{a+b} = T^b \\circ T^a`,
and it needs the state at the cut to be complete: a resumable cell re-emits
its input instead of zeros, so the run carries its own input and passes
with ``inject=False``.  That is why the segmented strategies do not
re-inject and the single-run one does.

A solver is a :class:`torch.nn.Module`, because a refresh and a halt head
own weights; a :class:`~discopy.neural.MapNN` registers it as a submodule.
Training loops are *not* here: a solver produces states, the caller decodes
them, computes a loss and steps an ordinary optimizer.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    Solver
    Iterate
    FixedPoint
    Refresh
    Recursion
    HaltHead
    ACT
"""

from __future__ import annotations

import torch


class Solver(torch.nn.Module):
    """
    A strategy for running a global interaction: from a state to the states
    a loss may look at.

    Subclasses implement :meth:`run`, which returns the final state and the
    list of checkpoints -- the states a caller decodes.  Every solver runs
    from and to the same flat state, so they are interchangeable at the
    same diagram.

    Parameters:
        rounds : The rounds of message passing per cycle.
        inject : Whether every round re-adds the initial messages, i.e.
                 whether the transition is :math:`\\sigma(\\Phi(s)) + i`.
    """
    def __init__(self, rounds: int = 1, inject: bool = False):
        super().__init__()
        self.rounds, self.inject = rounds, inject

    @property
    def depth(self) -> int:
        """ The rounds of message passing one run performs. """
        return self.rounds

    def run(self, interaction, state, deep: bool = False, **overrides):
        """
        Run the interaction from a state.

        Parameters:
            interaction : The compiled diagram to run.
            state : The flat initial messages.
            deep : Whether to keep every checkpoint rather than the last.
            overrides : Test-time compute, e.g. ``rounds``.

        Returns:
            The final state and the list of checkpoints.
        """
        raise NotImplementedError

    def forward(self, interaction, state, deep: bool = False, **overrides):
        """ See :meth:`run`. """
        return self.run(interaction, state, deep=deep, **overrides)


class Iterate(Solver):
    """
    Finite iteration: ``rounds`` rounds of message passing, and nothing
    more.

    ``T^n(s_0)`` is what it computes, whether or not the sequence goes
    anywhere.  With ``deep`` it returns the state after every round, which
    is what lets a loss supervise all of them inside one backward graph.

    Parameters:
        rounds : The number of rounds.
        inject : Whether the initial messages are re-added every round.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.map import interpret
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = interpret(from_relation(((1, ), (0, )), node),
    ...                  {peer: Dim(3), state: Dim(2)},
    ...                  {"cell": torch.nn.Identity()})
    >>> last, every = Iterate(rounds=4).run(pair, pair.zeros(1), deep=True)
    >>> len(every), torch.equal(last, every[-1])
    (4, True)
    """
    def __init__(self, rounds: int = 1, inject: bool = True):
        super().__init__(rounds, inject)

    def run(self, interaction, state, deep: bool = False, rounds: int = None):
        rounds = self.rounds if rounds is None else rounds
        if deep:
            every = interaction.advance(
                state, rounds, self.inject, per_round=True)
            return every[-1], list(every)
        state = interaction.advance(state, rounds, self.inject)
        return state, [state]


class FixedPoint(Solver):
    """
    Iteration towards a fixed point :math:`s^\\ast = T(s^\\ast)`, stopping
    when the residual :math:`\\|T(s) - s\\|_\\infty` falls under ``tol``.

    This is Picard iteration and it is the honest name for it: it stops
    early when the map happens to settle, and hits ``rounds`` when it does
    not.  Whether it settles is a property of the learned weights -- an
    untrained map generally does not -- so :meth:`residual` is reported
    rather than assumed, and no implicit-function theorem is invoked
    unless the caller asks for it.

    ``backward`` chooses the differentiation policy:

    * ``"last"`` -- iterate under ``no_grad`` and differentiate one final
      round from the detached limit.  This is the Jacobian-free one-step
      gradient used by deep-equilibrium models; it is the exact implicit
      gradient only in the limit where the Neumann series is dominated by
      its first term, and an approximation otherwise.
    * ``"full"`` -- backpropagate through every round actually run, i.e.
      unrolled backpropagation through the iteration.

    Parameters:
        rounds : The maximum number of rounds.
        tol : The residual to stop at, or ``None`` never to test -- each
              test costs one device synchronisation.
        inject : Whether the initial messages are re-added every round.
        backward : ``"last"`` or ``"full"``.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.map import interpret
    >>> from discopy.neural.signature import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> half = torch.nn.Linear(3, 3)
    >>> _ = half.weight.data.fill_(0.0), half.bias.data.fill_(0.0)
    >>> pair = interpret(from_relation(((1, ), (0, )), node),
    ...                  {peer: Dim(1), state: Dim(1)}, {"cell": half})
    >>> solved, _ = FixedPoint(rounds=32, tol=1e-8).run(pair, pair.zeros(2))
    >>> bool(pair.residual(solved).max() < 1e-8)
    True
    """
    def __init__(self, rounds: int = 32, tol: float = 1e-5,
                 inject: bool = False, backward: str = "last"):
        if backward not in ("last", "full"):
            raise ValueError(backward)
        super().__init__(rounds, inject)
        self.tol, self.backward = tol, backward

    def run(self, interaction, state, deep: bool = False, rounds: int = None,
            tol: float = None):
        rounds = self.rounds if rounds is None else rounds
        tol = self.tol if tol is None else tol
        free = rounds - 1 if self.backward == "last" else rounds
        every = []
        with torch.set_grad_enabled(
                self.backward == "full" and torch.is_grad_enabled()):
            for _ in range(max(free, 0)):
                found = interaction.advance(state, 1, self.inject)
                settled = tol is not None and bool(
                    (found - state).abs().amax() < tol)
                state = found
                if deep:
                    every.append(state)
                if settled:
                    break
        if self.backward == "last":
            state = interaction.advance(state.detach(), 1, self.inject)
            if deep:
                every.append(state)
        return state, every or [state]


class Refresh(torch.nn.Module):
    """
    One update of a trace from another, run between two cycles of a
    :class:`Recursion`.

    The target trace is a loop the generators read but never write: the
    solver refreshes it from the latent state, and every site is refreshed
    by the same shared module.

    Parameters:
        module : The recurrent cell, ``(latent, current) -> updated``.
        norm : The normalisation applied after it.
        source : The ``(generator name, role)`` of the latent state read.
        target : The ``(generator name, role)`` of the trace written.
    """
    def __init__(self, module, norm, source: tuple, target: tuple):
        super().__init__()
        self.module, self.norm = module, norm
        self.source, self.target = source, target

    def forward(self, interaction, state):
        """
        A copy of the state with the target trace refreshed.

        Parameters:
            interaction : The compiled diagram the state belongs to.
            state : The flat incoming messages.
        """
        latent = interaction.read(state, self.source)
        current = interaction.read(state, self.target)
        rows, sites, width = current.shape
        updated = self.norm(self.module(
            latent.reshape(rows * sites, -1),
            current.reshape(rows * sites, width)))
        return interaction.write(
            state, self.target, updated.reshape(rows, sites, width))


class Recursion(Solver):
    """
    Segmented recursion after :cite:t:`JolicoeurMartineau25`::

        for step in range(steps):          # detached from one another
            for cycle in range(cycles):    # only the last differentiated
                state = rounds of message passing
                state = refresh(state)

    A **cycle** denotes :math:`C_\\theta = R_\\theta \\circ T_\\theta^r`
    with :math:`R_\\theta` the refresh, and a **supervision step** denotes
    :math:`C_\\theta^{\\,\\text{cycles}}`.  Denotationally a step is that
    composite; *differentiably* only the last cycle is in the graph, which
    is what bounds activation memory to one segment.  With ``cycles > 1``
    no gradient reaches the state a step started from, nor the encoder that
    built it.

    Parameters:
        rounds : The rounds of message passing per cycle.
        cycles : The cycles per supervision step.
        steps : The supervision steps of a whole run.
        refresh : The :class:`Refresh` run at the end of every cycle, or
                  ``None`` for a recursion with no answer trace.
        inject : Whether the initial messages are re-added every round;
                 off, because a segmented run resumes from its own state
                 and :math:`T_{D,\\theta,i}` with a re-injected ``i`` is a
                 different transition from the one it carried.

    Example
    -------
    >>> Recursion(rounds=6, cycles=3, steps=8).depth
    144
    """
    def __init__(self, rounds: int = 1, cycles: int = 1, steps: int = 1,
                 refresh: Refresh = None, inject: bool = False):
        super().__init__(rounds, inject)
        self.cycles, self.steps = cycles, steps
        self.refresh = refresh

    @property
    def depth(self) -> int:
        """ The rounds of message passing one run performs. """
        return self.rounds * self.cycles * self.steps

    def cycle(self, interaction, state, rounds: int = None):
        """
        One cycle: ``rounds`` rounds of message passing, then one refresh.

        Parameters:
            interaction : The compiled diagram to run.
            state : The flat incoming messages.
            rounds : The rounds of this cycle, the solver's by default.
        """
        state = interaction.advance(
            state, self.rounds if rounds is None else rounds, self.inject)
        return state if self.refresh is None \
            else self.refresh(interaction, state)

    def step(self, interaction, state, cycles: int = None,
             grad: bool = True):
        """
        One supervision step: ``cycles - 1`` cycles without gradients and
        one with.

        Parameters:
            interaction : The compiled diagram to run.
            state : The flat incoming messages.
            cycles : The cycles of this step, the solver's by default.
            grad : Whether the last cycle is differentiated.
        """
        cycles = self.cycles if cycles is None else cycles
        with torch.no_grad():
            for _ in range(cycles - 1):
                state = self.cycle(interaction, state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            return self.cycle(interaction, state)

    def run(self, interaction, state, deep: bool = False, rounds: int = None,
            cycles: int = None, steps: int = None, grad: bool = True):
        if rounds is not None:
            raise ValueError(
                "a recursion changes its test-time compute with `steps`, "
                "not with `rounds`")
        every = []
        for _ in range(self.steps if steps is None else steps):
            state = Recursion.step(self, interaction, state, cycles,
                                   grad=grad)
            every = every + [state] if deep else [state]
            state = state.detach()
        return every[-1], every


class HaltHead(torch.nn.Linear):
    """
    The ``Q_head(y)`` of :cite:t:`JolicoeurMartineau25`: one logit read off
    a trace, trained to predict whether the answer it carries is already
    correct, so that an easy problem costs one or two supervision steps
    instead of all of them.

    Two shapes:

    * ``"mean"`` : one logit off the *mean* of the site embeddings, trained
      against whole-problem correctness -- the literal pooled translation
      of the paper's single ``q_hat``.
    * ``"softmin"`` : one logit *per* site, trained against per-site
      correctness and aggregated by a soft minimum,
      ``q = -logsumexp(-q_i)``.  The transformer original reads a slot that
      attention lets act as a soft minimum over the sites; a map has no
      such slot, so the aggregation is explicit.  It matches the target's
      conjunctive shape -- an answer is correct when *every* site is --
      where a mean dilutes one wrong site by one over their number, and it
      is conservative by construction, since ``q`` lower bounds the least
      confident site.

    The head is initialised to zero weights and a bias of ``-5``, the
    convention of the reference implementations, so that halting is
    initially rare and the recursion trains at full depth until the head
    has learned a signal -- and so that adding a halt head to a model draws
    exactly the random numbers the model without one draws, leaving the two
    bitwise identical under the same seed.

    Parameters:
        dim : The width of one site's embedding.
        kind : ``"mean"`` or ``"softmin"``.

    Example
    -------
    >>> head = HaltHead(4, "softmin")
    >>> bool((head.weight == 0).all()), float(head.bias)
    (True, -5.0)
    >>> round(float(head.logit(torch.zeros(2, 3)).max()), 6)
    -1.098612
    """

    def __init__(self, dim: int, kind: str = "mean"):
        if kind not in ("mean", "softmin"):
            raise ValueError(kind)
        with torch.random.fork_rng(devices=[]):
            super().__init__(dim, 1)
        self.kind = kind
        with torch.no_grad():
            self.weight.zero_()
            self.bias.fill_(-5.0)

    def read(self, answer):
        """
        The raw halt output: shape ``(rows, )`` for the mean head,
        ``(rows, sites)`` for the soft-minimum one.

        Parameters:
            answer : The site embeddings, ``(rows, sites, dim)``.
        """
        if self.kind == "mean":
            return self(answer.mean(dim=1)).squeeze(-1)
        return self(answer).squeeze(-1)

    def logit(self, halt):
        """
        The problem-level halt logit of shape ``(rows, )``: the raw logit
        of the mean head, the soft minimum of the per-site logits
        otherwise.

        Parameters:
            halt : The output of :meth:`read`.
        """
        if self.kind == "mean":
            return halt
        return -torch.logsumexp(-halt, dim=-1)

    def loss(self, halt, correct):
        """
        The binary cross-entropy of the head against correctness:
        problem-level for the mean head, per-site for the soft-minimum one.

        Parameters:
            halt : The output of :meth:`read`.
            correct : A boolean tensor of shape ``(rows, sites)``, whether
                      each site of the answer the head read is right.
        """
        target = correct.all(-1) if self.kind == "mean" else correct
        return torch.nn.functional.binary_cross_entropy_with_logits(
            halt, target.detach().to(halt.dtype))


class ACT(Recursion):
    """
    A recursion with a halt head, i.e. adaptive computation time.

    The inherited :meth:`~Recursion.run` is untouched, so fixed-compute
    evaluation of an ACT model is exactly fixed-compute evaluation of the
    recursion it was built from; :meth:`step` is the one that also reads
    the head.  The halting rule itself -- when to stop, what to do with a
    finished example -- is a training or evaluation policy and lives in the
    caller.

    Parameters:
        halt : The :class:`HaltHead`.
        detach : Whether the head reads a *detached* answer, so that its
                 loss trains the head alone and never touches the trunk.
                 Off by default, as in the paper.
        kwargs : Passed to :class:`Recursion`; ``refresh`` is required,
                 since the head reads the trace it refreshes.
    """
    def __init__(self, *args, halt: HaltHead, detach: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self.refresh is None:
            raise ValueError("an ACT solver needs a trace to halt on")
        # built after the recursion rather than among its parts, so that a
        # checkpoint of a model with a halt head is a checkpoint of one
        # without plus two tensors, in that order.
        self.halt = halt
        self.detach = detach

    def step(self, interaction, state, cycles: int = None,
             grad: bool = True):
        """
        One supervision step returning ``(state, answer, halt)``: the
        inherited step plus the trace it ended on and the halt output read
        off it.

        Parameters:
            interaction : The compiled diagram to run.
            state : The flat incoming messages.
            cycles : The cycles of this step, the solver's by default.
            grad : Whether the last cycle is differentiated.
        """
        cycles = self.cycles if cycles is None else cycles
        with torch.no_grad():
            for _ in range(cycles - 1):
                state = self.cycle(interaction, state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = self.cycle(interaction, state)
            answer = interaction.read(state, self.refresh.target)
            return state, answer, self.halt.read(
                answer.detach() if self.detach else answer)
