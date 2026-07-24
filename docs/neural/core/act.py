# -*- coding: utf-8 -*-

"""
Adaptive computation time (ACT) for the recursion solver, after the
tiny recursive model of :cite:t:`JolicoeurMartineau25`.

The paper's deep-supervision loop is, verbatim from its pseudocode::

    for step in range(N_supervision):
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss  = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))
        loss.backward(); opt.step(); opt.zero_grad()
        if q_hat > 0:   # early-stopping
            break

i.e. a single *halt* logit ``q_hat = Q_head(y)`` read off the current
answer, trained to predict whether that answer is already correct, and a
supervision loop that stops as soon as the model believes it -- so easy
puzzles cost one or two supervision steps instead of all ``N_sup``.

The ``break`` above is per *example*.  Batched training realises it as
**slot refill**, as in the reference implementations of the tiny
recursive model and of the hierarchical reasoning model it descends
from: every batch slot advances its own puzzle, and the moment a slot
halts -- its halt logit turns positive, or it hits the ``n_sup`` cap --
it is refilled with a fresh puzzle *within the same batch*.  The GPU
always steps a full batch of useful work, and the saving appears as
examples per second: at a mean halting depth of ``d``, one optimizer
step consumes ``batch_size / d`` puzzles instead of ``batch_size /
n_sup``.  The refill is computed entirely on the device -- a cumulative
sum turns the halt mask into ranks in the shuffled order -- so the
training loop stays free of host round-trips, exactly like the
``train_chunk`` loop it replaces.

Three pieces:

* :class:`ACTSolver` : :class:`core.solvers.RecursionSolver` plus the
  halt head.  Built with the same seed it has bitwise the same weights as
  a plain recursion solver -- the halt head is initialised to constants --
  and its inherited :meth:`~core.solvers.RecursionSolver.step` and
  ``forward`` are untouched, so fixed-compute evaluation of an ACT model
  is exactly fixed-compute evaluation of the recursion solver.
* :class:`ACTTrainer` : the slot-refill loop, one optimizer step per
  call unit, with the loss of the pseudocode.
* :func:`evaluate_act` : inference with the same early stopping, capped
  at a maximum number of supervision steps, reporting the mean depth.
"""

from __future__ import annotations

import numpy as np
import torch

from core.recipes import GRAD_CLIP
from core.solvers import RecursionSolver
from core.study import Widths
from core.train import decode, device_of


class ACTSolver(RecursionSolver):
    """
    The recursion solver with a halt head reading the answer embeddings
    ``y``, the
    ``Q_head(y)`` of the paper, in one of two shapes:

    * ``halt_head="mean"`` : one logit off the *mean* of the answer
      embeddings, trained against board-level correctness -- the literal
      pooled translation of the paper's single ``q_hat``.
    * ``halt_head="softmin"`` : one logit *per cell*, trained against
      per-cell correctness, aggregated by a soft minimum,
      ``q = -logsumexp(-q_i)``.  The transformer original reads its
      ``pred[:, 0]`` slot, which attention lets act as a soft minimum
      over the board; this map has no such slot, so the aggregation is
      explicit.  The soft minimum matches the target's conjunctive
      shape -- a board is correct when *every* cell is -- where a mean
      dilutes a single wrong cell by one over the number of cells; and it is conservative
      by construction, since ``q`` lower-bounds the least confident
      cell by up to the log of the number of cells, so ``q > 0`` already requires every
      cell to be confidently correct.  The per-cell targets also give
      the head a cells-fold denser, far less noisy training signal than
      the board-level binary.

    The head is initialised to zero weights and a bias of ``-5`` -- the
    convention of the reference implementations -- so that halting is
    initially rare (sigmoid of ``-5``) and the recursion trains at full
    depth until the head has learned a signal; and so that building an
    :class:`ACTSolver` draws exactly the random numbers a ``TRMSolver``
    draws, making the two bitwise identical under the same seed.

    Parameters:
        abstract : The factor-graph skeleton to interpret.
        widths : The widths of the model.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The maximum number of supervision steps, i.e. the halting
                cap of the paper's ``N_supervision``.
        n_classes : The number of classes a variable can take.
        halt_detach : Whether the halt head reads a *detached* answer, so
                      its loss trains the head alone and never touches the
                      trunk -- halting still works, at zero gradient cost
                      to the answer. Off by default, as in the paper.
        halt_head : ``"mean"`` (default) or ``"softmin"``, see above.
    """
    def __init__(self, abstract, widths: Widths, rounds: int,
                 cycles: int, n_sup: int, n_classes: int,
                 halt_detach: bool = False, halt_head: str = "mean"):
        assert halt_head in ("mean", "softmin"), halt_head
        super().__init__(abstract, widths, rounds, cycles, n_sup,
                         n_classes)
        self.halt_detach, self.halt_head = halt_detach, halt_head
        self.q_head = torch.nn.Linear(self.widths.y_dim, 1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def act_step(self, state, cycles: int = None, grad: bool = True):
        """
        One supervision step returning ``(state, logits, halt)``: the
        inherited :meth:`~core.solvers.RecursionSolver.step` -- ``T - 1``
        cycles without gradients, one with -- plus the halt logits read
        off the same differentiated answer that the readout decodes, as
        in the paper's ``Q_head(y)``: shape ``(batch, )`` for the mean
        head, ``(batch, n_cells)`` for the soft-minimum head.  Reduce
        with :meth:`halt_logit` for the halting decision and train with
        :meth:`halt_loss`.

        Parameters:
            state : The flat incoming messages.
            cycles : The cycles of this step, ``self.cycles`` by default.
            grad : Whether the last cycle is differentiated.
        """
        cycles = self.cycles if cycles is None else cycles
        with torch.no_grad():
            for _ in range(cycles - 1):
                state = self.cycle(state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = self.cycle(state)
            answer = self.router.read(state, self.answer_ports)[:, ::2]
            logits = self.readout_from(answer)
            read = answer.detach() if self.halt_detach else answer
            if self.halt_head == "mean":
                halt = self.q_head(read.mean(dim=1)).squeeze(-1)
            else:
                halt = self.q_head(read).squeeze(-1)
        return state, logits, halt

    def halt_logit(self, halt):
        """
        The board-level halt logit of shape ``(batch, )``: the raw logit
        of the mean head, the soft minimum ``-logsumexp(-q_i)`` of the
        per-cell logits otherwise.

        Parameters:
            halt : The halt output of :meth:`act_step`.
        """
        if self.halt_head == "mean":
            return halt
        return -torch.logsumexp(-halt, dim=-1)

    def halt_loss(self, halt, logits, targets):
        """
        The binary cross-entropy of the halt head against correctness:
        board-level -- ``(y_hat == y_true)`` of the paper -- for the
        mean head, per-cell for the soft-minimum head.

        Parameters:
            halt : The halt output of :meth:`act_step`.
            logits : The digit logits of the same step.
            targets : The 0-indexed solutions, ``(batch, n_cells)``.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        with torch.no_grad():
            cell_correct = logits.argmax(-1) == targets
            target = cell_correct.all(-1) if self.halt_head == "mean" \
                else cell_correct
        return bce(halt, target.to(halt.dtype))


class PuzzleStream:
    """
    A GPU-resident stream of training examples for the slot refill.

    Holds the whole training set on the device and a shuffled order;
    :meth:`take` maps a halt mask to the next examples of the stream
    without leaving the device, by ranking the halted slots with a
    cumulative sum.  The order is reshuffled -- from the ``numpy``
    generator, so runs are reproducible -- whenever the cursor has run
    past the end; until then indexing wraps around, so an example can
    only repeat once the whole set has been consumed.

    Parameters:
        clues : The training puzzles, ``(n, 81)`` long tensor on device.
        targets : The 0-indexed solutions, ``(n, 81)`` long on device.
        rng : The ``numpy`` generator behind the shuffles.
    """
    def __init__(self, clues: "torch.Tensor", targets: "torch.Tensor",
                 rng: "np.random.Generator"):
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
        Total puzzles drawn from the stream so far, across reshuffles: an
        absolute count comparable across trials regardless of their mean
        halting depth, unlike a count of optimizer steps. One host read
        of the cursor, so call it sparingly in a tight loop.
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

    def take(self, halted: "torch.Tensor"):
        """
        The next examples of the stream, one per *halted* slot: slot
        ``i`` with ``halted[i]`` receives the ``rank(i)``-th example
        after the cursor, where ``rank`` counts halted slots up to
        ``i``.  Entries of non-halted slots are arbitrary and meant to
        be discarded by the caller's ``torch.where``.

        Parameters:
            halted : Boolean mask of shape ``(batch, )``.
        """
        ranks = torch.cumsum(halted.long(), 0) - 1
        index = self.order[
            (self.cursor + ranks) % len(self.order)]
        self.cursor = self.cursor + halted.sum()
        return self.clues[index], self.targets[index]


class ACTTrainer:
    """
    The slot-refill training loop: a persistent batch of carries, one
    optimizer step per iteration, the loss of the paper's pseudocode.

    Each iteration runs :meth:`ACTSolver.act_step` on the whole batch,
    steps the optimizer on ``cross_entropy + binary_cross_entropy`` and
    then refills every halted slot -- halt logit positive, or ``n_sup``
    steps reached -- with a fresh puzzle from the stream, entirely on
    the device.  The carried state is detached (and cloned, for CUDA
    graphs) between iterations, exactly like the plain loop.

    The refilled states are built under ``no_grad``: with ``T > 1``
    cycles per step the first cycles are undifferentiated anyway, so the
    embedding and ``y0`` receive no gradient in either loop.

    Parameters:
        model : The :class:`ACTSolver` to train.
        stream : The :class:`PuzzleStream` feeding the refill.
        batch_size : The number of slots.
        grad_clip : The gradient-norm clip of every optimizer step.
        halt_weight : The weight of the halt loss in the total,
                      ``cross_entropy + halt_weight * binary_cross_entropy``
                      -- below 1 it reduces how much the halt head's
                      gradient competes with the answer's in the shared
                      trunk (moot under ``halt_detach``, where the halt
                      loss never reaches the trunk at any weight). The
                      reported ``q`` statistic stays unweighted.
        halt_threshold : The margin the halt logit must clear to stop a
                         slot, ``0`` for the paper's ``q > 0``; raise it
                         to halt more conservatively.
    """
    def __init__(self, model: ACTSolver, stream: PuzzleStream,
                 batch_size: int, grad_clip: float = GRAD_CLIP,
                 halt_weight: float = 1.0, halt_threshold: float = 0.0):
        assert model.cycles > 1, "T = 1 would differentiate the refill"
        self.model, self.stream = model, stream
        self.grad_clip = grad_clip
        self.halt_weight = halt_weight
        self.halt_threshold = halt_threshold
        device = stream.clues.device
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.clues, self.targets = stream.take(halted)
        with torch.no_grad():
            self.state = model.initial(self.clues)
        self.steps = torch.zeros(
            batch_size, dtype=torch.long, device=device)

    def _run_raw(self, optimizer, scheduler, ema, iterations: int) -> dict:
        """
        The core loop of ``iterations`` optimizer steps, returning the raw
        accumulated sums (not yet divided by ``iterations``/``halted``),
        so that :meth:`run` and :meth:`run_until` can normalize once over
        however many steps they cover. All statistics accumulate in device
        tensors and are read back once at the end, so the loop never
        blocks on the GPU mid-chunk.
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
        """ Turn raw accumulated sums over ``iterations`` steps into means. """
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
            supervision steps) and ``capped`` (the fraction that hit
            the ``n_sup`` cap rather than halting by ``q > 0``).
        """
        raw = self._run_raw(optimizer, scheduler, ema, iterations)
        return self._normalize(raw, iterations)

    def run_until(self, optimizer, scheduler, ema, target_consumed: int,
                 check_every: int = 50) -> dict:
        """
        Optimizer steps until the stream's cumulative consumed-puzzle count
        reaches ``target_consumed`` (an absolute count from
        :meth:`PuzzleStream.total_consumed`), the natural training and
        evaluation unit for ACT: the same puzzle budget spans a different
        number of optimizer steps depending on the model's mean halting
        depth, so pinning the budget to steps would make trials with
        different depths see different amounts of data.

        Progress is checked -- one host read of the stream cursor -- every
        ``check_every`` steps, so the loop still runs many optimizer steps
        between host round-trips; the run may overshoot ``target_consumed``
        by up to one ``check_every``-sized chunk.

        Parameters:
            optimizer : The optimizer, stepped once per iteration.
            scheduler : The learning-rate scheduler, stepped alongside.
            ema : The weight average updated after every step, or None.
            target_consumed : The absolute puzzle count to train up to.
            check_every : Optimizer steps per progress check.

        Returns:
            The same keys as :meth:`run`, pooled over however many steps
            were taken, plus ``iterations`` (the step count) and
            ``consumed`` (the puzzle count actually reached).
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


def evaluate_act(model: ACTSolver, split, max_sup: int = None,
                 batch_size: int = 2000, threshold: float = 0.0) -> dict:
    """
    Inference with the paper's early stopping: every puzzle runs until
    its halt logit clears ``threshold`` or ``max_sup`` supervision steps
    are reached, and is scored on the answer it halted with.  This
    changes the predictions relative to fixed-compute evaluation --
    stopping early is the point -- so both are worth reporting.

    Parameters:
        model : The trained :class:`ACTSolver`.
        split : The split to evaluate on.
        max_sup : The cap on supervision steps, ``model.n_sup`` default.
        batch_size : The evaluation batch size.
        threshold : The margin the halt logit must clear, ``0`` for the
                    paper's ``q > 0``; raise it to stop later.

    Returns:
        ``cell`` and ``board`` accuracy, and ``depth``, the mean number
        of supervision steps actually run per puzzle.
    """
    model.eval()
    max_sup = model.n_sup if max_sup is None else max_sup
    device = device_of(model)
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
            depth = torch.full((len(clues),), max_sup,
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
