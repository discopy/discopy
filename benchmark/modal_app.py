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

This is an *ephemeral* app: ``modal run`` spins the container up, runs
:func:`run_benchmark` once, and tears it down when the local entrypoint
returns -- no ``modal deploy`` / teardown step needed. The entrypoint runs
locally (wherever ``modal run`` was invoked, e.g. the GitHub runner) and
writes the returned JSON bytes straight to disk, so there is no Modal Volume
or separate download step: the result is on local disk as soon as the
command exits.
"""

import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON_VERSION = "3.14"  # matches the `python-version` matrix in benchmark.yml

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
def run_benchmark(bench_flags: str = "") -> bytes:
    """ Sync the project and run the benchmark suite, returning the raw
    ``--benchmark-json`` bytes. """
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
         "--benchmark-json=/tmp/bench.json"],
        cwd="/discopy", check=True, env=env)
    return pathlib.Path("/tmp/bench.json").read_bytes()


@app.local_entrypoint()
def main(output: str = "benchmark-results/bench.json", bench_flags: str = ""):
    """ Run the suite remotely and write the result where CI expects it. """
    data = run_benchmark.remote(bench_flags)
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")
