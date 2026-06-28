# -*- coding: utf-8 -*-

"""
Benchmark for diagram composition, reproducing the experiments of
arXiv:2105.09257 for both :class:`discopy.symmetric.Diagram` and
:class:`discopy.symmetric.Hypergraph`.

This module is deliberately kept outside of ``test/*/*.py`` so that it is
not collected by the default ``pytest`` run (see ``testpaths`` in
``pyproject.toml``). It is meant to be run explicitly, e.g.

    uv run pytest benchmark/ -v --log-cli-level=INFO --log-cli-format="%(message)s"

Each case is swept over a range of sizes (:data:`SIZES`) to expose its
scaling trend. Each case has its own time limit (:data:`CASE_MAX_SECONDS`),
so cheap-tailed cases run out to the largest sizes while quadratic ones stop
sooner; a predicted-setup guard (:data:`SETUP_CAP`) and the global
:data:`TIME_BUDGET` keep the whole sweep bounded, leaving the unreached
cells blank. The final ``test_zz_report`` renders every measurement as a
plain markdown table, with one row per case and one column per size.
Timings are emitted via :mod:`logging` at ``INFO`` level so they appear
inline in the pytest output without needing ``-s``.

Timings use ``time.process_time()`` (CPU time, immune to scheduling noise)
with the garbage collector disabled during the timed region, and report
the median of a few repetitions for the cheap cases.
"""

import gc
import logging
import math
import time
from statistics import median

from tabulate import tabulate

from discopy import compact, rigid
from discopy.symmetric import Ty, Box, Id, Diagram, Functor
from discopy.monoidal import Layer
from discopy.python import Function

# Sizes swept for every case, largest first dropped once a case gets too slow.
# A 1-2-5 series (and its multiples by powers of ten): evenly spaced on the
# log axis the scaling plot uses, so every case gets a comparable number of
# points across whatever range it reaches before hitting its cap.
SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
REPEATS = 3
TIME_BUDGET = 600          # total CPU budget for the whole sweep, in seconds

# Per-case cap on a single timed measurement (seconds): a case stops once a
# measurement crosses its cap. Cheap-tailed cases (the linear-ish Diagram and
# series builds) get a high cap so they run out to the largest sizes, while
# the quadratic Hypergraph builds and the spiral normalisation stop sooner.
# Tuned so the whole sweep fills as much of the table as it can while staying
# comfortably inside TIME_BUDGET.
CASE_MAX_SECONDS = {
    "k-fold tensor (Diagram)": 15.0,
    "k-fold tensor, 1 layer (Diagram)": 15.0,
    "k-fold tensor (Hypergraph)": 6.0,
    "staircase foliation (Diagram)": 6.0,
    "staircase to hypergraph (Hypergraph)": 6.0,
    "k-fold series (Diagram)": 15.0,
    "k-fold series (Hypergraph)": 8.0,
    "adder step (Diagram)": 12.0,
    "adder step (Hypergraph)": 12.0,
    "adder functor (Diagram)": 12.0,
    "spiral build (Diagram)": 6.0,
    "spiral build (Hypergraph)": 3.0,
    "spiral normal_form (Diagram)": 3.0,
    "spiral equality (Hypergraph)": 8.0,
    "transpose snake removal (Diagram)": 6.0,
    "transpose equality (Hypergraph)": 8.0,
}
DEFAULT_MAX_SECONDS = 4.0
# A few cases spend most of their time in *untimed* setup (building the adder,
# or the two spirals an equality check compares). Predict the next setup from
# the previous ones and skip a size whose setup would exceed the case's cap, so
# we never pay a runaway setup only to abandon the measurement right after. The
# adder is given a generous cap so its (incrementally built) construction can
# reach the sizes where the Hypergraph step overtakes the Diagram one.
DEFAULT_SETUP_CAP = 12.0
CASE_SETUP_CAP = {
    "adder step (Diagram)": 90.0,
    "adder step (Hypergraph)": 90.0,
    "adder functor (Diagram)": 90.0,
}

# case name -> {size: seconds}, in first-seen (i.e. report row) order.
_results: "dict[str, dict[int, float]]" = {}
_start_time = time.process_time()


def measure(fn):
    """ Median process-time of calling ``fn()``, with the GC disabled.

    Cheap calls (under a second) are repeated for stability; expensive ones
    are timed a single time so the sweep stays within its budget.
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.process_time()
        fn()
        first = time.process_time() - start
        times = [first]
        if first <= 1.0:
            for _ in range(REPEATS - 1):
                start = time.process_time()
                fn()
                times.append(time.process_time() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return median(times)


def _extrapolate(history, n, default_exponent):
    """ Predict the cost at size ``n`` from ``(size, cost)`` history.

    Fits a local power law ``cost ~ size ** p`` to the last two points (the
    benchmarks are polynomial in size), falling back to ``default_exponent``
    when only one point is known.
    """
    prev_n, prev_cost = history[-1]
    if len(history) >= 2:
        (n0, c0), (n1, c1) = history[-2], history[-1]
        if c0 > 0 and c1 > 0 and n1 > n0:
            exponent = min(max(
                math.log(c1 / c0) / math.log(n1 / n0), 1.0), 4.0)
        else:
            exponent = default_exponent
    else:
        exponent = default_exponent
    return prev_cost * (n / prev_n) ** exponent


def run_scaling(name, prepare, sizes=SIZES):
    """ Sweep ``name`` over ``sizes``, recording one row of the report.

    ``prepare(n)`` performs any *untimed* setup for size ``n`` and returns a
    zero-argument callable that is then timed. Scaling stops for this case
    once the measured time crosses the case's entry in
    :data:`CASE_MAX_SECONDS`, the predicted setup crosses the case's setup cap
    (:data:`CASE_SETUP_CAP`, else :data:`DEFAULT_SETUP_CAP`), or the global
    :data:`TIME_BUDGET` is about to be exhausted.
    """
    row = _results.setdefault(name, {})
    cap = CASE_MAX_SECONDS.get(name, DEFAULT_MAX_SECONDS)
    setup_cap = CASE_SETUP_CAP.get(name, DEFAULT_SETUP_CAP)
    setups, timings = [], []
    for n in sizes:
        elapsed = time.process_time() - _start_time
        # Stop if even an optimistic next measurement risks the global budget.
        # The 0.75 factor leaves a reserve for one in-flight tail (and for the
        # prediction undershooting) so a slow runner cannot trip the assert.
        if timings and \
                elapsed + _extrapolate(timings, n, 2.0) > 0.75 * TIME_BUDGET:
            break
        # Skip a size whose untimed setup is predicted to blow up.
        if setups and _extrapolate(setups, n, 3.0) > setup_cap:
            break
        setup_start = time.process_time()
        thunk = prepare(n)
        setups.append((n, time.process_time() - setup_start))
        duration = measure(thunk)
        timings.append((n, duration))
        row[n] = duration
        logging.info("%-32s n=%-5d %9.4f s", name, n, duration)
        if duration > cap:
            break


def repeated(op, box, k):
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    if k % 2:
        result = op(result, box)
    return result


def single_layer_tensor(box, k):
    """ The ``k``-fold tensor of ``box`` as a *single* :class:`Layer`.

    A monoidal ``Diagram`` is a list of layers, and tensoring two diagrams
    pads every layer of each with the wires of the other, so building a
    ``k``-fold tensor by repeated ``@`` rebuilds the layer list over and over.
    The same morphism is just one layer with the ``k`` boxes side by side
    (interleaved with empty types), which sidesteps that overhead entirely --
    the residual cost is only the type concatenation in the ``Layer``
    constructor.
    """
    empty = box.dom[:0]
    layer = Layer(empty, box, empty, *([box, empty] * (k - 1)))
    return Diagram((layer,), layer.dom, layer.cod)


def staircase(box, k):
    """ The ``k``-fold tensor of ``box`` as a staircase of ``k`` layers.

    Tensoring two diagrams stacks their layers, so ``box @ box @ ...`` puts
    each box in its own layer at an increasing offset -- the same morphism as
    :func:`single_layer_tensor`, but spread over ``k`` layers instead of one.
    """
    g = box
    for _ in range(k - 1):
        g = g @ box
    return g


def test_not_gate_tensor():
    bit = Ty('bit')
    box = Box('NOT', bit, bit)
    run_scaling(
        "k-fold tensor (Diagram)",
        lambda n: lambda: repeated(lambda a, b: a.tensor(b), box, n))
    run_scaling(
        "k-fold tensor, 1 layer (Diagram)",
        lambda n: lambda: single_layer_tensor(box, n))
    hbox = box.to_hypergraph()
    run_scaling(
        "k-fold tensor (Hypergraph)",
        lambda n: lambda: repeated(lambda a, b: a.tensor(b), hbox, n))


def test_foliation():
    # The inverse of single_layer_tensor: take the k-layer staircase and pack
    # it back into one layer. The Diagram does this with foliation, repeatedly
    # merging adjacent layers (super-linear). The Hypergraph carries no layer
    # structure at all, so to_hypergraph reaches the maximally-parallel form
    # directly -- the staircase and the single layer have the same hypergraph.
    bit = Ty('bit')
    box = Box('NOT', bit, bit)

    def prepare_foliation(n):
        st = staircase(box, n)
        return lambda: st.foliation()
    run_scaling("staircase foliation (Diagram)", prepare_foliation)

    def prepare_to_hypergraph(n):
        st = staircase(box, n)
        return lambda: st.to_hypergraph()
    run_scaling("staircase to hypergraph (Hypergraph)", prepare_to_hypergraph)


def test_not_gate_series():
    bit = Ty('bit')
    box = Box('NOT', bit, bit)
    run_scaling(
        "k-fold series (Diagram)",
        lambda n: lambda: repeated(lambda a, b: a.then(b), box, n))
    hbox = box.to_hypergraph()
    run_scaling(
        "k-fold series (Hypergraph)",
        lambda n: lambda: repeated(lambda a, b: a.then(b), hbox, n))


def permutation(factory, xs, dom):
    """ A permutation arrow built from swaps, generic over the category.

    Mirrors :meth:`symmetric.Diagram.permutation`, but uses only ``id``,
    ``swap``, ``tensor`` and ``then`` -- the operations every monoidal
    category shares -- so the very same code builds a Diagram (when
    ``factory`` is a ``symmetric.Box``/``Diagram``) or a Hypergraph directly.
    """
    if len(dom) <= 1:
        return factory.id(dom)
    i = xs[0]
    head = factory.swap(dom[:i], dom[i:i + 1]).tensor(factory.id(dom[i + 1:]))
    tail = factory.id(dom[i:i + 1]).tensor(permutation(
        factory, [x - 1 if x > i else x for x in xs[1:]],
        dom[:i] + dom[i + 1:]))
    return head.then(tail)


def adder_step(full_adder, adder, k):
    """ One incremental ripple-carry step: adder(k) -> adder(k + 1).

    Parameterised by the addition box ``full_adder``: a ``symmetric.Box``
    grows a Diagram-valued adder, a ``Hypergraph`` grows a hypergraph-valued
    one. Everything else (the identities, the swaps inside the reordering
    permutations) is taken from the box's own category, ``type(full_adder)``,
    so the Diagram and Hypergraph adders are built from one single recipe --
    the only difference is which category the ``full_adder`` lives in.
    """
    factory = type(full_adder)
    bit = full_adder.dom[:1]
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    step = adder.tensor(factory.id(bit @ bit))
    step = step.then(permutation(factory, reorder1, step.cod))
    step = step.then(factory.id(bit ** k).tensor(full_adder))
    return step.then(permutation(factory, reorder2, step.cod))


def incremental_adder(full_adder, measure=None):
    """ A ``prepare(n)`` that grows one adder across the whole size sweep.

    Instead of rebuilding an ``n``-cell adder from scratch at every size
    (which is itself the dominant, super-quadratic cost), keep the running
    adder between calls and only extend it from its current size up to ``n``.
    The untimed setup at each size is then just the incremental extension, so
    the row reaches noticeably larger sizes within the same budget.

    By default the returned thunk times one further ``adder(n) -> adder(n + 1)``
    step. Pass ``measure(adder, k)`` to time something else built from the
    running adder instead (e.g. applying a functor to it).
    """
    if measure is None:
        measure = lambda adder, k: lambda: adder_step(full_adder, adder, k)
    state = {"adder": full_adder, "size": 1}

    def prepare(n):
        while state["size"] < n:
            state["adder"] = adder_step(
                full_adder, state["adder"], state["size"])
            state["size"] += 1
        return measure(state["adder"], state["size"])
    return prepare


def test_adder():
    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)
    run_scaling("adder step (Diagram)", incremental_adder(full_adder))

    full_adder_hg = full_adder.to_hypergraph()
    run_scaling(
        "adder step (Hypergraph)", incremental_adder(full_adder_hg))


def test_adder_functor():
    # The adder is the natural case for benchmarking *functor application*: a
    # Python-valued functor sends the full-adder box to an actual full adder
    # (sum and carry bits), so applying it to adder(k) compiles the whole
    # ripple-carry diagram down to one Python function adding its input bits.
    # We first check that function is correct -- the diagram is a carry-save
    # accumulator, so reading its k + 1 outputs with weights [1, 2, 2, ..., 2]
    # recovers the number of ones fed in -- then time the functor application
    # as the adder grows.
    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)

    def full_adder_function(a, b, carry_in):
        return a ^ b ^ carry_in, (a & b) | (carry_in & (a ^ b))

    functor = Functor(
        ob={bit: int}, ar={full_adder: full_adder_function}, cod=Function)

    def carry_save_value(outputs):
        # The first output is the running sum bit (weight 1), the rest are
        # carries (weight 2 each); their weighted sum is the integer encoded.
        return outputs[0] + 2 * sum(outputs[1:])

    import itertools
    adder = full_adder
    for k in range(1, 5):  # exhaustively check the small adders are correct
        compiled = functor(adder)
        for bits in itertools.product((0, 1), repeat=2 * k + 1):
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = adder_step(full_adder, adder, k)

    run_scaling("adder functor (Diagram)", incremental_adder(
        full_adder, measure=lambda adder, k: lambda: functor(adder)))


def make_spiral(n_cups):
    """ The diagram of arXiv:1804.07832, built with symmetric boxes. """
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result, unit, counit


def test_spiral():
    run_scaling(
        "spiral build (Diagram)", lambda n: lambda: make_spiral(n)[0])

    def prepare_build_hg(n):
        spiral = make_spiral(n)[0]
        return lambda: spiral.to_hypergraph()
    run_scaling("spiral build (Hypergraph)", prepare_build_hg)

    def prepare_normal_form(n):
        spiral = make_spiral(n)[0]
        return lambda: spiral.normal_form()
    run_scaling("spiral normal_form (Diagram)", prepare_normal_form)

    def prepare_equality(n):
        # Two independent builds of the same closed spiral: the equality
        # check must decide they are isomorphic. The spiral is closed (empty
        # boundary), hence *not* boundary-connected and not monogamous, so
        # this exercises the networkx graph-isomorphism (VF2) fallback rather
        # than the linear-time boundary-rooted path (see test_transpose for
        # the latter).
        left = make_spiral(n)[0].to_hypergraph()
        right = make_spiral(n)[0].to_hypergraph()
        assert left == right
        return lambda: left == right
    run_scaling("spiral equality (Hypergraph)", prepare_equality)


def identity_snake(category, x):
    """ An identity on ``x`` routed through a right-then-left transpose snake.

    Equal to ``Id(x)`` but built from a cap and a cup, so composing ``n`` of
    them after a box wraps it in ``n`` snakes while keeping the diagram a
    *constant* width (unlike nesting transposes, which would make it ``O(n)``
    wide and so quadratic to convert).
    """
    return category.Id(x).transpose(left=False).transpose(left=True)


def with_snakes(f, snake, n):
    """ Compose ``f`` with ``n`` identity snakes, equal to ``f`` but cluttered.
    """
    g = f
    for _ in range(n):
        g = g >> snake
    return g


def test_transpose():
    # An endomorphism cluttered with n identity snakes, recognised as equal to
    # f in two completely different ways. The Diagram side uses *rigid*
    # diagrams, whose normal_form genuinely yanks the snakes back to f (rigid
    # has left/right duals and a real snake-removal normal form), and which is
    # super-linear. The Hypergraph side uses compact diagrams -- compact has no
    # separate normal form, the hypergraph *is* the compact-closed normal form
    # -- and times the full to_hypergraph + equality, i.e. the snakes are
    # absorbed during construction and equality with the bare box is then by
    # the linear-time boundary-rooted canonical form (the snaked diagram is
    # monogamous and boundary-connected, so the fast path, not the VF2 fallback
    # the closed spiral above hits). Construction is included in the timing, so
    # this is the honest linear-vs-super-linear comparison, not just the O(1)
    # equality of two already-built hypergraphs.
    rigid_x = rigid.Ty('x')
    rigid_f = rigid.Box('f', rigid_x, rigid_x)
    rigid_snake = identity_snake(rigid, rigid_x)
    assert with_snakes(rigid_f, rigid_snake, 1).normal_form() == rigid_f

    def prepare_snake_removal(n):
        g = with_snakes(rigid_f, rigid_snake, n)
        return lambda: g.normal_form()
    run_scaling("transpose snake removal (Diagram)", prepare_snake_removal)

    compact_x = compact.Ty('x')
    compact_f = compact.Box('f', compact_x, compact_x)
    compact_snake = identity_snake(compact, compact_x)
    bare = compact_f.to_hypergraph()

    def prepare_equality(n):
        # The diagram is built in setup; the timed call includes to_hypergraph
        # (the snake-absorbing construction) plus the equality, so this is not
        # a cheat that excludes the cost of building the hypergraph.
        g = with_snakes(compact_f, compact_snake, n)
        assert g.to_hypergraph() == bare
        return lambda: g.to_hypergraph() == bare
    run_scaling("transpose equality (Hypergraph)", prepare_equality)


def write_report(table_markdown):
    """ Save the table, the raw timings and a log-log scaling plot.

    Writes ``results.md``, ``results.json`` and ``scaling.png`` into the
    directory named by the ``BENCHMARK_OUTPUT`` environment variable (default
    ``benchmark-results``), which the CI workflow uploads as an artifact. The
    plot splits the Diagram and Hypergraph cases into two panels so their
    slopes (i.e. their complexities) can be read off and compared.
    """
    import json
    import os

    output = os.environ.get("BENCHMARK_OUTPUT", "benchmark-results")
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "results.md"), "w") as file:
        file.write(table_markdown + "\n")
    with open(os.path.join(output, "results.json"), "w") as file:
        json.dump(_results, file, indent=2, sort_keys=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for name, row in _results.items():
        axis = axes[1] if "Hypergraph" in name else axes[0]
        xs = sorted(row)
        axis.plot(xs, [row[x] for x in xs], marker="o", label=name)
    for axis, title in zip(axes, ["Diagram", "Hypergraph"]):
        axis.set(xscale="log", yscale="log", xlabel="size $n$", title=title)
        axis.grid(True, which="both", linestyle=":", linewidth=.5)
        axis.legend(fontsize="small")
    axes[0].set_ylabel("median CPU time (s)")
    figure.suptitle("Composition benchmark scaling (arXiv:2105.09257)")
    figure.tight_layout()
    figure.savefig(os.path.join(output, "scaling.png"), dpi=120)
    plt.close(figure)
    logging.info("wrote results.{md,json} and scaling.png to %s/", output)


def test_zz_report():
    """ Render the scaling sweep as a plain markdown table. Runs last.

    Named ``test_zz_*`` so it runs after every other test has populated
    ``_results``. Run with ``--log-cli-level=INFO`` to see the table, which is
    also saved with the raw timings and a plot by :func:`write_report`.
    """
    sizes = sorted({n for row in _results.values() for n in row})
    table = [
        [name] + [f"{row[n]:.4f}" if n in row else "" for n in sizes]
        for name, row in _results.items()]
    table_markdown = tabulate(
        table, headers=["case", *sizes], tablefmt="github",
        colalign=["left", *["right"] * len(sizes)], disable_numparse=True)
    logging.info("\n%s", table_markdown)
    write_report(table_markdown)

    total = time.process_time() - _start_time
    assert total < TIME_BUDGET, \
        f"Benchmark took {total:.1f}s, over the {TIME_BUDGET}s budget."
