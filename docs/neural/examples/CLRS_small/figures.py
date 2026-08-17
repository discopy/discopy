# -*- coding: utf-8 -*-

"""
The figures of Part 2, drawn from the artefacts and from nothing else.

    python figures.py                       # every report that exists
    python figures.py --algorithms bfs

Three of them, and each answers a question a scalar cannot:

* :func:`tracking` -- the ``argmax``-over-the-nodes hint probes against
  the *fraction* of the trajectory elapsed, so that the two splits share
  an axis.  **Executor or shortcut**, which is the question every reading
  of Part 3 depends on: a model that tracks the algorithm and then loses
  it has an iteration to stabilize, and a model that never tracks it at
  this size does not, however well it scores on the output.
* :func:`hints` -- one score per hint probe per step of the trajectory, in
  distribution and out of it.  *Where* the imitation comes apart: a curve
  that starts high and falls is a model that tracks the algorithm and then
  loses it, a curve that is flat and low is a probe never learned, and a
  whole panel that is uniformly mediocre is what a misaligned
  checkpoint-to-step mapping looks like.
* :func:`residuals` -- the residual after every round, run past the trained
  depth, against the round at which the *algorithm* stops changing.  Part
  3's H2 instrument, drawn now so that it has a baseline: a fixed-point
  iteration should settle and a sequential scan need not, and the
  period-two oscillation of the two clocks is visible in both.  The
  overlay is what makes it an instrument -- "the learned map settles where
  the algorithm does" is a claim about the distance between two things on
  one axis, so they are drawn on one axis.

Nothing here recomputes a number; a figure that disagrees with its
artefact is a bug in the figure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import ALGORITHMS, ARTIFACTS, FIGURES, FULL, Budget  # noqa: E402
from dataset import kind  # noqa: E402
from discopy.neural.cells import POOL  # noqa: E402


def read(algorithm: str, budget: Budget = FULL) -> dict:
    """ The report of one algorithm, or ``None`` if it has none yet. """
    path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
    return json.loads(path.read_text()) if path.exists() else None


def hints(reports: dict, path=None):
    """
    One panel per algorithm: the score of every hint probe along the
    trajectory, in distribution (solid) and out of it (dashed).

    A scalar probe is marked in the legend, because CLRS scores one with a
    **mean squared error** and every other type with an F1 or an accuracy:
    `bellman_ford`'s ``d`` at 0.02 is its best curve and its neighbours at
    0.02 would be their worst, and one axis carrying both is worth exactly
    one label saying so.

    Parameters:
        reports : Per algorithm, what :func:`~evaluate.report` wrote.
        path : Where to save, ``figures/hints.png`` by default.
    """
    figure, axes = grid(len(reports))
    for axis, (algorithm, found) in zip(axes, reports.items()):
        curves = found["seeds"][0]["hints"]
        colours = {}
        for split, style in (("val", "-"), ("ood", "--")):
            for name, curve in curves[split].items():
                colours.setdefault(name, f"C{len(colours)}")
                scalar = kind(algorithm, name)[1] == "scalar"
                axis.plot(range(1, 1 + len(curve)), curve, style,
                          color=colours[name],
                          label=None if split != "val" else
                          f"{name} (MSE)" if scalar else name)
        axis.set_title(algorithm, fontsize=10)
        axis.set_xlabel("algorithm step")
        axis.set_ylabel("hint score (solid n=16, dashed n=64)", fontsize=8)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=7, loc="lower left")
    figure.tight_layout()
    figure.savefig(path or FIGURES / "hints.png", dpi=150)
    plt.close(figure)


def tracking(reports: dict, path=None):
    """
    One panel per algorithm: the score of every ``argmax``-over-the-nodes
    hint probe against the **fraction of the trajectory elapsed**, in
    distribution (solid) and out of it (dashed).

    This is :func:`hints` with the one change that makes it decide
    something.  Drawn against the absolute step index the two splits live
    on different axes -- fifteen steps against sixty-three -- so a curve
    that tracks and then drifts and a curve that never tracks at all are
    two shapes at two scales and the eye cannot compare them.  Against
    the *fraction* elapsed they are the same axis, and the two shapes are
    the question Part 3 rests on: a dashed curve that leaves the solid
    one's level and falls is a model executing the algorithm and losing
    it, and one that starts low and stays low is a model that never
    executed it at this size, whatever its output score says.

    Only the ``pointer`` and ``mask_one`` probes are drawn, because they
    are the heads whose candidate set is the graph and therefore the only
    ones a change of size reaches; a ``mask`` is a sigmoid per node and
    generalizes here, which :func:`~evaluate.tracking_table` records.

    Parameters:
        reports : Per algorithm, what :func:`~evaluate.report` wrote.
        path : Where to save, ``figures/tracking.png`` by default.
    """
    figure, axes = grid(len(reports))
    for axis, (algorithm, found) in zip(axes, reports.items()):
        curves = found["seeds"][0]["hints"]
        colours = {}
        for split, style in (("val", "-"), ("ood", "--")):
            for name, curve in curves[split].items():
                if kind(algorithm, name)[1] not in ("pointer", "mask_one"):
                    continue
                colours.setdefault(name, f"C{len(colours)}")
                elapsed = [(one + 1) / len(curve) for one in range(len(curve))]
                axis.plot(elapsed, curve, style, color=colours[name],
                          label=name if split == "val" else None)
        axis.set_title(algorithm, fontsize=10)
        axis.set_xlabel("fraction of the trajectory")
        axis.set_ylabel("hint score (solid n=16, dashed n=64)", fontsize=8)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=7, loc="lower left")
    figure.tight_layout()
    figure.savefig(path or FIGURES / "tracking.png", dpi=150)
    plt.close(figure)


def grid(panels: int, width: int = 4):
    """
    A figure of ``panels`` axes wrapped at ``width`` per row, and the axes
    flat and trimmed to the panels asked for.

    Eight algorithms in one row is eight panels a page wide, which is a
    figure nobody reads at the size it is published; the same eight in two
    rows of four is one anybody can.

    Example
    -------
    >>> figure, axes = grid(8)
    >>> len(axes), figure.axes == list(axes)
    (8, True)
    >>> plt.close(figure)
    """
    rows = -(-panels // width)
    figure, axes = plt.subplots(
        rows, min(panels, width), squeeze=False,
        figsize=(4 * min(panels, width), 3.2 * rows))
    for axis in axes.flat[panels:]:
        figure.delaxes(axis)
    return figure, list(axes.flat)[:panels]


def residuals(reports: dict, path=None):
    """
    One panel per algorithm: the residual after every round, run to the
    deepest point of the sweep, **with the algorithm's own convergence on
    the same axes**.

    The green band is where the sampled executions stop changing -- the
    10th to 90th percentile of :func:`~evaluate.settling`, on the same
    round axis, with its median as a solid rule.  That overlay is the
    whole point of the figure and the reason it is not two figures: a
    falling residual is a fact about a learned map, and it says something
    about *alignment* only when it is read against where the thing being
    imitated settles.  Both are drawn per split, because a trajectory at
    ``n = 64`` is longer than one at ``n = 16`` and the two bands are not
    in the same place.

    Parameters:
        reports : Per algorithm, what :func:`~evaluate.report` wrote.
        path : Where to save, ``figures/residuals.png`` by default.
    """
    figure, axes = grid(len(reports))
    for axis, (algorithm, found) in zip(axes, reports.items()):
        curve = found["seeds"][0]["residual_curve"]
        settles = found.get("settles") or {}
        for split, style, colour in (("val", "-", "C0"), ("ood", "--", "C1")):
            size = 16 if split == "val" else 64
            axis.plot(range(1, 1 + len(curve[split])), curve[split], style,
                      color=colour, label=f"{split} (n={size})")
            settle = settles.get(split)
            if settle is None:
                continue
            axis.axvspan(settle["low"], settle["high"], color=colour,
                         alpha=0.12, lw=0)
            axis.axvline(settle["median"], color=colour, lw=1.1, alpha=0.55,
                         label=f"{split}: algorithm settles")
        trained = len(curve["val"]) / curve["factor"]
        axis.axvline(trained, color="grey", lw=0.8, ls=":")
        axis.set_title(f"{algorithm}", fontsize=10)
        axis.set_xlabel("round (dotted: trained depth)")
        axis.set_ylabel(r"$\|T(s)-s\|_\infty$", fontsize=8)
        axis.set_yscale("log")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path or FIGURES / "residuals.png", dpi=150)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    parser.add_argument("--pool", choices=sorted(POOL), default=None,
                        help="the campaign to draw, `max` being the primary")
    arguments = parser.parse_args(argv)
    budget = FULL if arguments.pool is None \
        else replace(FULL, pool=arguments.pool)
    found = {name: read(name, budget) for name in arguments.algorithms}
    found = {name: one for name, one in found.items() if one is not None}
    if not found:
        parser.error("no report to draw; run evaluate.py first")
    tracking(found)
    hints(found)
    residuals(found)
    print(f"  {len(found)} algorithms -> {FIGURES}/tracking.png, "
          "hints.png, residuals.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
