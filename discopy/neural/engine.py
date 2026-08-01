# -*- coding: utf-8 -*-

"""
Running a map: one nested loop, one schedule.

Message passing over a closed map is the execution formula, and composing
rounds is all a solver ever does.  What distinguishes the architectures of
the literature is *when* a loss looks at the messages and *whether* the
input is re-injected -- not the wiring, and not the cell.  A
:class:`Schedule` names exactly those choices, and :class:`Engine` runs
them with the same three nested loops::

    for step in range(steps):            # detached segments
        for cycle in range(cycles):      # the last one differentiated
            state = rounds of message passing

* a geometry-of-interaction factor graph is ``Schedule(16, 1, 1,
  inject=True, supervise="round")`` -- one differentiated run, a loss on
  every round;
* a segmented recursion after :cite:t:`JolicoeurMartineau25` is
  ``Schedule(6, 3, 8, inject=False, supervise="step")`` -- gradients only
  through the last cycle of each of eight detached segments.

Resuming is a property of the formalism, ``F^(a+b) = F^b . F^a``, and it
needs the state at the cut to be complete: a resumable site re-emits its
input instead of zeros, so the run carries its own input and passes with
``inject=False``.

:class:`RecursionEngine` adds the answer trace -- a loop the sites read but
never write, refreshed from the latent state between cycles -- and
:class:`ACTEngine` adds the halt head of the same paper, so adaptive
computation time rides on top of *any* skeleton rather than one model.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:

    Schedule
    Engine
    RecursionEngine
    HaltHead
    ACTEngine
    PuzzleStream
    ACTTrainer
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from discopy.neural.functor import Interpretation, interpret

#: The default gradient-norm clip of an optimizer step.
GRAD_CLIP = 1.0


@dataclass(frozen=True)
class Schedule:
    """
    When a loss looks at the messages, and whether the input is
    re-injected.

    Parameters:
        rounds : The rounds of message passing per cycle.
        cycles : The cycles per supervision step, the first ``cycles - 1``
                 of them undifferentiated.
        steps : The supervision steps, detached from one another.
        inject : Whether to re-add the initial messages every round rather
                 than just the first.
        supervise : ``"round"`` to read a checkpoint off every round of one
                    differentiated run, ``"step"`` to read one off every
                    detached segment.

    Example
    -------
    >>> Schedule(16, 1, 1, inject=True, supervise="round").depth
    16
    >>> Schedule(6, 3, 8, inject=False, supervise="step").depth
    144
    """

    rounds: int = 1
    cycles: int = 1
    steps: int = 1
    inject: bool = True
    supervise: str = "round"

    def __post_init__(self):
        if self.supervise not in ("round", "step"):
            raise ValueError(self.supervise)

    @property
    def depth(self) -> int:
        """ The rounds of message passing one forward pass runs. """
        return self.rounds * self.cycles * self.steps

    def at(self, rounds: int = None, cycles: int = None,
           steps: int = None) -> Schedule:
        """
        The same schedule at a different test-time compute, ``None``
        keeping what is already there.

        Parameters:
            rounds : The rounds per cycle.
            cycles : The cycles per supervision step.
            steps : The supervision steps.
        """
        return replace(
            self, rounds=self.rounds if rounds is None else rounds,
            cycles=self.cycles if cycles is None else cycles,
            steps=self.steps if steps is None else steps)


class Engine(torch.nn.Module):
    """
    A skeleton, an interpretation of it and a schedule, as one torch
    module.

    Parameters:
        skeleton : The abstract wiring to interpret.
        ob : The :class:`~discopy.neural.Dim` each atomic role carries.
        parts : The modules of the engine, as an *ordered* mapping from
                attribute name to a zero-argument factory.
        sites : The attribute name of the module filling each box name.
        schedule : When a loss looks at the messages.
        clue : The ``(box name, role)`` of the input trace.
        state : The ``(box name, role)`` the readout decodes.
        n_classes : The number of classes the readout emits.
        encoder : The attribute name of the module embedding the input.
        decoder : The attribute name of the readout.

    Note
    ----
    ``parts`` is a mapping of *factories*, not of modules, and its order is
    load-bearing: every constructor draws from the global generator, so
    building the readout before the cells gives a different model under the
    same seed.  Passing factories is what lets the engine own that order
    instead of leaving it to the caller to remember.
    """

    def __init__(self, skeleton, ob: Mapping, parts: Mapping,
                 sites: Mapping, schedule: Schedule,
                 clue: tuple, state: tuple, n_classes: int,
                 encoder: str = "embedding", decoder: str = "readout"):
        super().__init__()
        for name, factory in parts.items():
            setattr(self, name, factory())
        self.schedule, self.n = schedule, n_classes
        self.encoder_name, self.decoder_name = encoder, decoder
        self.skeleton, self.ob = skeleton, dict(ob)
        self.wiring = interpret(Interpretation(ob, {
            box: getattr(self, attribute)
            for box, attribute in sites.items()}), skeleton)
        self.grid = self.wiring.cmap
        self.router = self.wiring.router
        self.clue_ports = self.wiring.ports[clue]
        self.state_ports = self.wiring.heads[state]
        self.answer_ports: tuple = ()
        self.clue_copies = len(self.clue_ports) // len(
            self.wiring.heads[clue])
        self.n_cells = len(self.wiring.heads[clue])
        self.cells = self.grid.as_network().module

    @property
    def encoder(self) -> torch.nn.Module:
        """ The module embedding the input. """
        return getattr(self, self.encoder_name)

    @property
    def decoder(self) -> torch.nn.Module:
        """ The readout. """
        return getattr(self, self.decoder_name)

    @property
    def outer_loop(self) -> bool:
        """ Whether the loss is applied once per detached segment. """
        return self.schedule.supervise == "step"

    @property
    def rounds(self) -> int:
        """ The rounds of message passing per cycle. """
        return self.schedule.rounds

    @property
    def cycles(self) -> int:
        """ The cycles per supervision step. """
        return self.schedule.cycles

    @property
    def n_sup(self) -> int:
        """ The supervision steps of one forward pass. """
        return self.schedule.steps

    @property
    def n_wires(self) -> int:
        """ The number of wires of the map. """
        return self.wiring.n_wires

    def compile_cells(self, **kwargs) -> Engine:
        """
        Compile the round step of the map with ``torch.compile``; see
        :meth:`discopy.neural.CMap.compile`.  Message passing on these maps
        is launch-bound -- many small kernels per round -- so this is a
        several-fold wall-clock speedup on a GPU at identical numerics up
        to rounding error.  Named ``compile_cells`` so as not to shadow
        ``torch.nn.Module.compile``.
        """
        self.grid.compile(**kwargs)
        return self

    # --- the state ---------------------------------------------------------

    def initial(self, x):
        """
        The initial per-port messages: the embedded input on every copy of
        the input trace, zeros everywhere else.

        Parameters:
            x : The input, of shape ``(batch, n_cells)``.
        """
        assert x.shape[1] == self.n_cells, "wrong number of cells"
        embedded = self.encoder(x)
        assert embedded.shape[:2] == (len(x), self.n_cells), \
            "bad embedding shape"
        flat = torch.zeros(len(x), self.router.total,
                           dtype=embedded.dtype, device=embedded.device)
        return self.router.write(
            flat, self.clue_ports,
            embedded.repeat_interleave(self.clue_copies, dim=1))

    def advance(self, state, rounds: int, per_round: bool = False):
        """
        ``rounds`` rounds of message passing from a flat state, reading
        back the flat incoming messages of the next round.

        Parameters:
            state : The flat incoming messages.
            rounds : The rounds to run.
            per_round : Whether to read a checkpoint off every round.
        """
        return self.cells(init=state, n_rounds=rounds,
                          inject=self.schedule.inject,
                          return_rounds=per_round, return_flat=True)

    def refresh(self, state):
        """ What happens between two cycles; nothing, without an answer. """
        return state

    def decode(self, state):
        """
        Class logits of shape ``(batch, n_cells, n)`` read off a flat
        state.

        Parameters:
            state : The flat incoming messages.
        """
        logits = self.decoder(self.router.read(state, self.state_ports))
        assert logits.shape[1:] == (self.n_cells, self.n), "bad logits shape"
        return logits

    # --- the loop ----------------------------------------------------------

    def run(self, state, schedule: Schedule, deep: bool = False,
            grad: bool = True, detach: bool = False):
        """
        The nested loop: ``steps`` supervision steps of ``cycles`` cycles
        of ``rounds`` rounds, returning the final state and the logits at
        every supervised checkpoint.

        Only the last cycle of a step is differentiated, which is what
        bounds the activation memory of a deep recursion to one segment;
        with ``cycles == 1`` that is every cycle, and the whole run is one
        backward graph.

        Parameters:
            state : The flat incoming messages to start from.
            schedule : The schedule to run.
            deep : Whether to keep every checkpoint or only the last.
            grad : Whether the differentiated cycle is differentiated.
            detach : Whether to cut the graph between supervision steps.
        """
        every, per_round = [], deep and schedule.supervise == "round"
        for _ in range(schedule.steps):
            for cycle in range(schedule.cycles):
                with torch.set_grad_enabled(
                        grad and cycle == schedule.cycles - 1
                        and torch.is_grad_enabled()):
                    emitted = self.advance(state, schedule.rounds, per_round)
                    state = emitted[-1] if per_round else emitted
                    state = self.refresh(state)
                    if per_round:
                        every += [self.decode(flat) for flat in emitted]
                    elif cycle == schedule.cycles - 1:
                        every.append(self.decode(state))
            if detach:
                state = state.detach()
        return state, every

    def forward(self, x, deep: bool = False, rounds: int = None,
                cycles: int = None, n_sup: int = None):
        """
        The logits of a whole run, at every supervised checkpoint when
        ``deep``, at the last one otherwise.

        Parameters:
            x : The input, of shape ``(batch, n_cells)``.
            deep : Whether to return every checkpoint.
            rounds : The rounds per cycle, the schedule's by default.
            cycles : The cycles per supervision step.
            n_sup : The supervision steps.
        """
        schedule = self.schedule.at(rounds, cycles, n_sup)
        _, every = self.run(self.initial(x), schedule, deep=deep,
                            grad=not self.outer_loop,
                            detach=self.outer_loop)
        return every if deep else every[-1]


class RecursionEngine(Engine):
    """
    An engine whose sites carry an *answer* trace they read but never
    write, refreshed from their latent state between two cycles.

    The map is the plain one plus one loop per site; what changes is the
    schedule, which is segmented, and the readout, which decodes the
    answer rather than the latent state.

    Parameters:
        answer : The ``(box name, role)`` of the answer trace.
        refresh : The attribute name of the module refreshing the answer.
        answer_norm : The attribute name of the normalisation after it.
        initial : The attribute name of the learned initial answer.
        kwargs : Passed to :class:`Engine`.
    """

    def __init__(self, *args, answer: tuple, refresh: str = "answer",
                 answer_norm: str = "answer_norm", initial: str = "y0",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_ports = self.wiring.ports[answer]
        self.answer_heads = self.wiring.heads[answer]
        self.answer_copies = len(self.answer_ports) // len(self.answer_heads)
        self.refresh_name, self.answer_norm_name = refresh, answer_norm
        self.initial_name = initial
        self.y_dim = self.router.widths[self.answer_heads[0]]
        self.state_width = self.router.widths[self.state_ports[0]]

    def initial(self, x):
        """ The shared initial messages plus the learned initial answer. """
        flat = super().initial(x)
        answer = getattr(self, self.initial_name).expand(
            len(x), self.n_cells, self.y_dim)
        return self.router.write(
            flat, self.answer_ports,
            answer.repeat_interleave(self.answer_copies, dim=1))

    def refresh(self, state):
        """
        One refresh of the answer from the latent state, i.e. what a cycle
        ends with.

        Parameters:
            state : The flat incoming messages.
        """
        latent = self.router.read(state, self.state_ports)
        answer = self.router.read(state, self.answer_heads)
        assert latent.shape[1:] == (self.n_cells, self.state_width)
        assert answer.shape[1:] == (self.n_cells, self.y_dim)
        updated = getattr(self, self.answer_norm_name)(
            getattr(self, self.refresh_name)(
                latent.reshape(-1, self.state_width),
                answer.reshape(-1, self.y_dim))
            ).reshape(len(state), self.n_cells, self.y_dim)
        return self.router.write(
            state, self.answer_ports,
            updated.repeat_interleave(self.answer_copies, dim=1))

    def decode(self, state):
        """ Class logits read off the answer rather than the state. """
        logits = self.decoder(self.router.read(state, self.answer_heads))
        assert logits.shape[1:] == (self.n_cells, self.n), "bad logits shape"
        return logits

    def cycle(self, state, rounds: int = None):
        """
        One macro-step: ``rounds`` rounds of resumable message passing,
        then one refresh of the answer.

        Parameters:
            state : The flat incoming messages.
            rounds : The rounds of this cycle, the schedule's by default.
        """
        assert state.shape[-1] == self.router.total, "bad state width"
        return self.refresh(self.advance(
            state, self.schedule.rounds if rounds is None else rounds))

    def step(self, state, cycles: int = None, grad: bool = True):
        """
        One supervision step: ``cycles - 1`` cycles without gradients and
        one with, returning the new state and the logits of the answer.

        Parameters:
            state : The flat incoming messages.
            cycles : The cycles of this step, the schedule's by default.
            grad : Whether the last cycle is differentiated.
        """
        state, every = self.run(
            state, self.schedule.at(cycles=cycles, steps=1), grad=grad)
        return state, every[-1]


class HaltHead(torch.nn.Linear):
    """
    The ``Q_head(y)`` of :cite:t:`JolicoeurMartineau25`: one logit read off
    the answer, trained to predict whether that answer is already correct,
    so that an easy problem costs one or two supervision steps instead of
    all of them.

    Two shapes:

    * ``"mean"`` : one logit off the *mean* of the answer embeddings,
      trained against whole-problem correctness -- the literal pooled
      translation of the paper's single ``q_hat``.
    * ``"softmin"`` : one logit *per* site, trained against per-site
      correctness and aggregated by a soft minimum,
      ``q = -logsumexp(-q_i)``.  The transformer original reads a slot
      that attention lets act as a soft minimum over the board; a map has
      no such slot, so the aggregation is explicit.  It matches the
      target's conjunctive shape -- a solution is correct when *every*
      site is -- where a mean dilutes one wrong site by one over their
      number, and it is conservative by construction, since ``q`` lower
      bounds the least confident site.  The per-site targets also give the
      head a far denser, less noisy signal than one binary per problem.

    The head is initialised to zero weights and a bias of ``-5``, the
    convention of the reference implementations, so that halting is
    initially rare and the recursion trains at full depth until the head
    has learned a signal -- and so that building an engine with a halt head
    draws exactly the random numbers one without draws, leaving the two
    bitwise identical under the same seed.

    Parameters:
        dim : The width of an answer embedding.
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
        super().__init__(dim, 1)
        self.kind = kind
        with torch.no_grad():
            self.weight.zero_()
            self.bias.fill_(-5.0)

    def read(self, answer):
        """
        The raw halt output: shape ``(batch, )`` for the mean head,
        ``(batch, n_cells)`` for the soft-minimum one.

        Parameters:
            answer : The answer embeddings, ``(batch, n_cells, dim)``.
        """
        if self.kind == "mean":
            return self(answer.mean(dim=1)).squeeze(-1)
        return self(answer).squeeze(-1)

    def logit(self, halt):
        """
        The problem-level halt logit of shape ``(batch, )``: the raw logit
        of the mean head, the soft minimum of the per-site logits
        otherwise.

        Parameters:
            halt : The output of :meth:`read`.
        """
        if self.kind == "mean":
            return halt
        return -torch.logsumexp(-halt, dim=-1)

    def loss(self, halt, logits, targets):
        """
        The binary cross-entropy of the head against correctness:
        problem-level for the mean head, per-site for the soft-minimum
        one.

        Parameters:
            halt : The output of :meth:`read`.
            logits : The class logits of the same step.
            targets : The 0-indexed solutions, ``(batch, n_cells)``.
        """
        with torch.no_grad():
            correct = logits.argmax(-1) == targets
            target = correct.all(-1) if self.kind == "mean" else correct
        return torch.nn.functional.binary_cross_entropy_with_logits(
            halt, target.to(halt.dtype))


class ACTEngine(RecursionEngine):
    """
    A recursion engine with a halt head, i.e. adaptive computation time.

    Its inherited :meth:`~RecursionEngine.step` and ``forward`` are
    untouched, so fixed-compute evaluation of an ACT model is exactly
    fixed-compute evaluation of the recursion it was built from.

    Parameters:
        halt : A zero-argument factory of the :class:`HaltHead`, called
               after the engine is built.
        halt_detach : Whether the head reads a *detached* answer, so its
                      loss trains the head alone and never touches the
                      trunk.  Off by default, as in the paper.
        kwargs : Passed to :class:`RecursionEngine`.
    """

    def __init__(self, *args, halt, halt_detach: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # built after the engine rather than among its ``parts``, so that a
        # checkpoint of an engine with a halt head is a checkpoint of one
        # without plus two tensors, in that order.
        self.q_head = halt()
        self.halt_detach = halt_detach

    def act_step(self, state, cycles: int = None, grad: bool = True):
        """
        One supervision step returning ``(state, logits, halt)``: the
        inherited :meth:`~RecursionEngine.step` plus the halt output read
        off the same differentiated answer that the readout decodes.

        Parameters:
            state : The flat incoming messages.
            cycles : The cycles of this step, the schedule's by default.
            grad : Whether the last cycle is differentiated.
        """
        cycles = self.schedule.cycles if cycles is None else cycles
        with torch.no_grad():
            for _ in range(cycles - 1):
                state = self.cycle(state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = self.cycle(state)
            answer = self.router.read(state, self.answer_heads)
            logits = self.decoder(answer)
            read = answer.detach() if self.halt_detach else answer
            halt = self.q_head.read(read)
        return state, logits, halt

    def halt_logit(self, halt):
        """ The problem-level halt logit; see :meth:`HaltHead.logit`. """
        return self.q_head.logit(halt)

    def halt_loss(self, halt, logits, targets):
        """ The halt head's loss; see :meth:`HaltHead.loss`. """
        return self.q_head.loss(halt, logits, targets)


class PuzzleStream:
    """
    A device-resident stream of training examples for the slot refill.

    Holds the whole training set on the device and a shuffled order;
    :meth:`take` maps a halt mask to the next examples of the stream
    without leaving the device, by ranking the halted slots with a
    cumulative sum.  The order is reshuffled -- from the ``numpy``
    generator, so runs are reproducible -- whenever the cursor has run past
    the end; until then indexing wraps around, so an example can only
    repeat once the whole set has been consumed.

    Parameters:
        clues : The training inputs, on the device.
        targets : The 0-indexed solutions, on the device.
        rng : The ``numpy`` generator behind the shuffles.
    """
    def __init__(self, clues, targets, rng):
        assert len(clues) == len(targets), "clues and targets disagree"
        self.clues, self.targets, self.rng = clues, targets, rng
        self.consumed_before = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = torch.as_tensor(
            self.rng.permutation(len(self.clues)), device=self.clues.device)
        self.cursor = torch.zeros(
            (), dtype=torch.long, device=self.clues.device)

    def total_consumed(self) -> int:
        """
        Total examples drawn from the stream so far, across reshuffles: an
        absolute count comparable across trials regardless of their mean
        halting depth, unlike a count of optimizer steps.  One host read of
        the cursor, so call it sparingly in a tight loop.
        """
        return self.consumed_before + int(self.cursor)

    def reshuffle_if_spent(self) -> None:
        """
        Reshuffle when the cursor has passed the end of the order.  Call
        between chunks, where the loop synchronises anyway; the one
        ``.item()`` here is the only host read of the stream.
        """
        if int(self.cursor) >= len(self.order):
            self.consumed_before += len(self.order)
            self._reshuffle()

    def take(self, halted):
        """
        The next examples of the stream, one per *halted* slot: slot ``i``
        with ``halted[i]`` receives the ``rank(i)``-th example after the
        cursor, where ``rank`` counts halted slots up to ``i``.  Entries of
        non-halted slots are arbitrary and meant to be discarded by the
        caller's ``torch.where``.

        Parameters:
            halted : Boolean mask of shape ``(batch, )``.
        """
        ranks = torch.cumsum(halted.long(), 0) - 1
        index = self.order[(self.cursor + ranks) % len(self.order)]
        self.cursor = self.cursor + halted.sum()
        return self.clues[index], self.targets[index]


class ACTTrainer:
    """
    The slot-refill training loop: a persistent batch of carries, one
    optimizer step per iteration, the loss of the paper's pseudocode.

    The paper's per-example ``break`` is realised as **slot refill**, as in
    the reference implementations: every batch slot advances its own
    problem, and the moment a slot halts -- its halt logit turns positive,
    or it hits the cap -- it is refilled with a fresh problem *within the
    same batch*.  The device always steps a full batch of useful work, and
    the saving appears as examples per second: at a mean halting depth of
    ``d``, one optimizer step consumes ``batch_size / d`` problems instead
    of ``batch_size / n_sup``.  The refill is computed entirely on the
    device -- a cumulative sum turns the halt mask into ranks in the
    shuffled order -- so the loop stays free of host round-trips.

    The refilled states are built under ``no_grad``: with more than one
    cycle per step the first cycles are undifferentiated anyway, so the
    encoder and the initial answer receive no gradient in either loop.

    Parameters:
        model : The :class:`ACTEngine` to train.
        stream : The :class:`PuzzleStream` feeding the refill.
        batch_size : The number of slots.
        grad_clip : The gradient-norm clip of every optimizer step.
        halt_weight : The weight of the halt loss in the total -- below 1
                      it reduces how much the head's gradient competes with
                      the answer's in the shared trunk (moot under
                      ``halt_detach``).  The reported ``q`` stays
                      unweighted.
        halt_threshold : The margin the halt logit must clear to stop a
                         slot, ``0`` for the paper's ``q > 0``.
    """
    def __init__(self, model: ACTEngine, stream: PuzzleStream,
                 batch_size: int, grad_clip: float = GRAD_CLIP,
                 halt_weight: float = 1.0, halt_threshold: float = 0.0):
        assert model.cycles > 1, "one cycle would differentiate the refill"
        self.model, self.stream = model, stream
        self.grad_clip = grad_clip
        self.halt_weight = halt_weight
        self.halt_threshold = halt_threshold
        device = stream.clues.device
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.clues, self.targets = stream.take(halted)
        with torch.no_grad():
            self.state = model.initial(self.clues)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _run_raw(self, optimizer, scheduler, ema, iterations: int) -> dict:
        """
        The core loop of ``iterations`` optimizer steps, returning the raw
        accumulated sums, so that :meth:`run` and :meth:`run_until` can
        normalize once over however many steps they cover.  All statistics
        accumulate in device tensors and are read back once at the end, so
        the loop never blocks on the device mid-chunk.
        """
        model, stream = self.model, self.stream
        ce = torch.nn.functional.cross_entropy
        model.train()
        device = stream.clues.device
        zeros = torch.zeros((), device=device)
        totals = {key: zeros.clone() for key in (
            "loss", "ce", "q", "halted", "solved", "depth", "capped")}
        step_zero = torch.zeros_like(self.steps)

        for _ in range(iterations):
            torch.compiler.cudagraph_mark_step_begin()
            state, logits, halt = model.act_step(self.state)
            ce_loss = ce(logits.reshape(-1, model.n),
                         self.targets.reshape(-1))
            with torch.no_grad():
                correct = (logits.argmax(-1) == self.targets).all(-1)
            q_loss = model.halt_loss(halt, logits, self.targets)
            loss = ce_loss + self.halt_weight * q_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                steps = self.steps + 1
                wants_halt = model.halt_logit(halt.detach()) \
                    > self.halt_threshold
                capped = steps >= model.n_sup
                halted = wants_halt | capped

                totals["loss"] += loss.detach()
                totals["ce"] += ce_loss.detach()
                totals["q"] += q_loss.detach()
                totals["halted"] += halted.sum()
                totals["solved"] += (halted & correct).sum()
                totals["depth"] += (steps * halted).sum()
                totals["capped"] += (capped & ~wants_halt).sum()

                new_clues, new_targets = stream.take(halted)
                mask = halted.unsqueeze(-1)
                self.clues = torch.where(mask, new_clues, self.clues)
                self.targets = torch.where(mask, new_targets, self.targets)
                fresh = model.initial(self.clues)
                # clone, not just detach: under reduce-overhead the state
                # is a graph output whose buffer the next replay reuses.
                self.state = torch.where(
                    mask, fresh, state.detach().clone())
                self.steps = torch.where(halted, step_zero, steps)

        return {key: value.item() for key, value in totals.items()}

    @staticmethod
    def _normalize(raw: dict, iterations: int) -> dict:
        """ Turn raw sums over ``iterations`` steps into means. """
        halted_count = max(raw["halted"], 1.0)
        return {
            "loss": raw["loss"] / iterations,
            "ce": raw["ce"] / iterations,
            "q": raw["q"] / iterations,
            "halted": int(raw["halted"]),
            "solved": raw["solved"] / halted_count,
            "depth": raw["depth"] / halted_count,
            "capped": raw["capped"] / halted_count}

    def run(self, optimizer, scheduler, ema, iterations: int) -> dict:
        """
        ``iterations`` optimizer steps of ACT training.

        Parameters:
            optimizer : The optimizer, stepped once per iteration.
            scheduler : The learning-rate scheduler, stepped alongside.
            ema : The weight average updated after every step, or None.
            iterations : The number of optimizer steps to run.

        Returns:
            ``loss`` (mean total), ``ce`` and ``q`` (its two terms),
            ``halted`` (examples finished), ``solved`` (of which whose
            final answer was correct), ``depth`` (their mean number of
            supervision steps) and ``capped`` (the fraction that hit the
            cap rather than halting by ``q > 0``).
        """
        return self._normalize(
            self._run_raw(optimizer, scheduler, ema, iterations), iterations)

    def run_until(self, optimizer, scheduler, ema, target_consumed: int,
                  check_every: int = 50) -> dict:
        """
        Optimizer steps until the stream's cumulative consumed count
        reaches ``target_consumed``, the natural training and evaluation
        unit for ACT: the same budget spans a different number of optimizer
        steps depending on the model's mean halting depth, so pinning the
        budget to steps would make trials with different depths see
        different amounts of data.

        Progress is checked -- one host read of the stream cursor -- every
        ``check_every`` steps, so the loop still runs many optimizer steps
        between host round-trips; the run may overshoot by up to one chunk.

        Parameters:
            optimizer : The optimizer, stepped once per iteration.
            scheduler : The learning-rate scheduler, stepped alongside.
            ema : The weight average updated after every step, or None.
            target_consumed : The absolute count to train up to.
            check_every : Optimizer steps per progress check.

        Returns:
            The same keys as :meth:`run`, pooled over however many steps
            were taken, plus ``iterations`` and ``consumed``.
        """
        pooled = {key: 0.0 for key in (
            "loss", "ce", "q", "halted", "solved", "depth", "capped")}
        total_iterations = 0
        while self.stream.total_consumed() < target_consumed:
            raw = self._run_raw(optimizer, scheduler, ema, check_every)
            self.stream.reshuffle_if_spent()
            for key in pooled:
                pooled[key] += raw[key]
            total_iterations += check_every
        result = self._normalize(pooled, max(total_iterations, 1))
        result["iterations"] = total_iterations
        result["consumed"] = self.stream.total_consumed()
        return result


def evaluate_act(model: ACTEngine, split, decode, max_sup: int = None,
                 batch_size: int = 2000, threshold: float = 0.0) -> dict:
    """
    Inference with the paper's early stopping: every problem runs until its
    halt logit clears ``threshold`` or ``max_sup`` supervision steps are
    reached, and is scored on the answer it halted with.  This changes the
    predictions relative to fixed-compute evaluation -- stopping early is
    the point -- so both are worth reporting.

    Parameters:
        model : The trained :class:`ACTEngine`.
        split : The split to evaluate on, with ``puzzles`` and
                ``solutions`` arrays.
        decode : The task's decode rule, ``(logits, clues) -> classes``.
        max_sup : The cap on supervision steps, the schedule's by default.
        batch_size : The evaluation batch size.
        threshold : The margin the halt logit must clear.

    Returns:
        ``cell`` and ``board`` accuracy, and ``depth``, the mean number of
        supervision steps actually run per problem.
    """
    import numpy as np
    model.eval()
    max_sup = model.n_sup if max_sup is None else max_sup
    device = next(model.parameters()).device
    correct, solved, depths = [], [], []
    with torch.no_grad():
        for start in range(0, len(split), batch_size):
            stop = start + batch_size
            clues = torch.as_tensor(
                split.puzzles[start:stop], dtype=torch.long, device=device)
            target = torch.as_tensor(
                split.solutions[start:stop], dtype=torch.long, device=device)
            torch.compiler.cudagraph_mark_step_begin()
            state = model.initial(clues)
            halted = torch.zeros(
                len(clues), dtype=torch.bool, device=device)
            final = torch.zeros(
                len(clues), model.n_cells, model.n, device=device)
            depth = torch.full((len(clues), ), max_sup,
                               dtype=torch.long, device=device)
            for t in range(1, max_sup + 1):
                torch.compiler.cudagraph_mark_step_begin()
                state, logits, halt = model.act_step(state, grad=False)
                state = state.detach().clone()
                newly = ~halted & (
                    (model.halt_logit(halt) > threshold) | (t == max_sup))
                final = torch.where(
                    newly[:, None, None], logits.to(final.dtype), final)
                depth = torch.where(
                    newly, torch.full_like(depth, t), depth)
                halted |= newly
                if bool(halted.all()):
                    break
            matches = decode(final, clues) == target
            correct.append(matches.float().mean(1).cpu().numpy())
            solved.append(matches.all(1).cpu().numpy())
            depths.append(depth.cpu().numpy())
    correct = np.concatenate(correct)
    return {"cell": float(correct.mean()),
            "board": float(np.concatenate(solved).mean()),
            "depth": float(np.concatenate(depths).mean())}
