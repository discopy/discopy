# -*- coding: utf-8 -*-

"""
Run the composition benchmark on a dedicated `Modal <https://modal.com>`_
container instead of the shared GitHub Actions runner.

The CI runner is roughly 7-8x slower than typical dev/Modal hardware for this
suite (observed: the *base-size-only* sweep alone took ~12 minutes there, see
PR #385/#346), which leaves no room for the heavy ``bench:full`` tail under
the workflow's 15-minute timeout. Rather than trim the suite, move it to
hardware that isn't shared, noisy, or time-boxed by CI concerns:

    uv run modal run benchmark/modal_app.py \\
        --bench-flags bench:full --output benchmark-results/bench.json

This is an *ephemeral* app: ``modal run`` spins the containers up, runs
:func:`run_group` once per group (see below), and tears them down when the
local entrypoint returns -- no ``modal deploy`` / teardown step needed. The
entrypoint runs locally (wherever ``modal run`` was invoked, e.g. the GitHub
runner) and writes the merged JSON bytes straight to disk, so there is no
Modal Volume or separate download step: the result is on local disk as soon
as the command exits.

The suite doesn't parallelize *within* a process -- pytest-benchmark disables
all timing outright if it detects pytest-xdist ("Benchmarks cannot be
performed reliably in a parallelized environment"), and the workload itself
is single-threaded/GIL-bound pure Python anyway. Real parallelism instead
comes from fanning out to separate *containers* via ``.starmap()``, each with
its own dedicated cores -- unlike xdist workers, they never share a core, so
timings stay valid. Cases are grouped by measured cost (not just by theme) so
each container gets a roughly even share; the heaviest single case
(``transpose_equality_hypergraph``, ~O(n^3) in ``to_hypergraph``) gets its own
group since nothing else is large enough to pair with it.
"""

import json
import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON_VERSION = "3.14"  # matches the `python-version` matrix in benchmark.yml

# (name, -k expression) -- disjoint and covers every test in test_composition.py.
# Rough measured full-sweep cost per group (see benchmark/test_composition.py
# comments/PR discussion): transpose ~4min, spiral+adder ~3.5min, the rest ~1min.
GROUPS = [
    ("transpose_equality_hypergraph", "transpose_equality_hypergraph"),
    ("snakes+adder+spiral",
     "(transpose or adder or spiral) and not transpose_equality_hypergraph"),
    ("tensor+series+staircase", "tensor or series or foliation or staircase"),
]

app = modal.App("discopy-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    # Baked into the image layer (cached) so it isn't re-downloaded on every
    # run -- only when this line or an earlier layer changes.
    .run_commands(f"uv python install {PYTHON_VERSION}")
    .add_local_dir(
        REPO_ROOT,
        remote_path="/discopy",
        ignore=[
            ".git", ".venv", ".uv-cache", ".pytest_cache",
            "**/__pycache__", "*.egg-info", "build", "dist",
            "benchmark-results", "docs/_build",
        ],
    )
)


@app.function(image=image, cpu=4, memory=4096, timeout=1800)
def run_group(bench_flags: str, k_expr: str) -> bytes:
    """ Sync the project and run one group's tests, returning the raw
    ``--benchmark-json`` bytes for just that group. """
    import os
    import subprocess

    # No .git in the mount (see `ignore=` above), so setuptools-scm can't
    # infer a version from tags; pin one directly instead.
    env = {
        **os.environ, "BENCH_FLAGS": bench_flags,
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0",
    }
    subprocess.run(
        ["uv", "sync", "--locked", "--python", PYTHON_VERSION, "--group", "dev"],
        cwd="/discopy", check=True, env=env)
    subprocess.run(
        ["uv", "run", "--python", PYTHON_VERSION, "pytest", "benchmark/", "-v",
         "--ignore=benchmark/modal_app.py", "-k", k_expr,
         "--benchmark-json=/tmp/bench.json"],
        cwd="/discopy", check=True, env=env)
    return pathlib.Path("/tmp/bench.json").read_bytes()


@app.local_entrypoint()
def main(output: str = "benchmark-results/bench.json", bench_flags: str = ""):
    """ Run each group remotely in parallel (one dedicated container per
    group -- see the module docstring for why this is safe and xdist isn't),
    merge the per-group JSON blobs, and write the result where CI expects
    it. """
    blobs = list(run_group.starmap(
        [(bench_flags, k_expr) for _, k_expr in GROUPS]))
    merged = {"benchmarks": []}
    for (name, _), blob in zip(GROUPS, blobs):
        group_result = json.loads(blob)
        print(f"{name}: {len(group_result['benchmarks'])} benchmarks")
        merged.setdefault("machine_info", group_result.get("machine_info"))
        merged["benchmarks"] += group_result["benchmarks"]

    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged))
    print(f"wrote {path} ({len(merged['benchmarks'])} benchmarks total)")
